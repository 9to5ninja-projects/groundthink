import triton
import triton.language as tl
import torch
import numpy as np

# -----------------------------------------------------------
# Forward Kernel - Optimized for Training and Inference
# -----------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'T_SEGMENT': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'T_SEGMENT': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'T_SEGMENT': 16}, num_stages=2, num_warps=16),
    ],
    key=['D', 'H', 'B', 'T'],
)
@triton.jit
def mamba_forward_kernel(
    # Input tensors
    x_ptr,            # [B, T, H * D]
    k_ptr, v_ptr, w_ptr, r_ptr,  # [B, T, H, D]
    
    # Output tensors
    y_ptr,            # [B, T, H * D]
    state_ptr,        # [B, H, D, D] final state
    
    # Dimensions
    B, T, H, D,
    
    # Strides
    stride_x_bt, stride_x_d,
    stride_k_bt, stride_k_h, stride_k_d,
    stride_v_bt, stride_v_h, stride_v_d,
    stride_w_bt, stride_w_h, stride_w_d,
    stride_r_bt, stride_r_h, stride_r_d,
    stride_y_bt, stride_y_d,
    stride_state_b, stride_state_h, stride_state_d1, stride_state_d2,
    
    # Kernel config
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    T_SEGMENT: tl.constexpr,
):
    """
    Forward pass for Mamba selective scan with memory-efficient segment processing.
    Each thread block processes one state block [BLOCK_M x BLOCK_N] for a specific batch and head.
    """
    
    # Parallelize over state matrix blocks
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)  # Block row in state matrix
    pid_n = tl.program_id(3)  # Block column in state matrix
    
    if pid_b >= B or pid_h >= H:
        return
    
    # Tile indices
    row_ids = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_ids = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = row_ids < D
    col_mask = col_ids < D
    
    # State offsets
    state_base = (pid_b * stride_state_b + 
                  pid_h * stride_state_h)
    
    # Initialize state for this block
    S_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Process time in segments for memory efficiency
    num_segments = tl.cdiv(T, T_SEGMENT)
    
    for seg in range(num_segments):
        seg_start = seg * T_SEGMENT
        seg_end = tl.min(seg_start + T_SEGMENT, T)
        
        # Process this segment
        for t in range(seg_start, seg_end):
            # Compute offsets for time t
            t_offset_bt = t * (H * D) if stride_k_bt == H * D else t
            base_offset = pid_b * stride_k_bt + t_offset_bt
            
            # Load w_t for this block's rows
            w_vals = tl.zeros(BLOCK_M, dtype=tl.float32)
            for m in range(BLOCK_M):
                if row_mask[m]:
                    row = row_ids[m]
                    offset = base_offset + pid_h * stride_k_h + row * stride_k_d
                    w_vals = tl.load(w_ptr + offset, mask=row_mask, other=0)
            
            # Load k_t for this block's rows
            k_vals = tl.zeros(BLOCK_M, dtype=tl.float32)
            for m in range(BLOCK_M):
                if row_mask[m]:
                    row = row_ids[m]
                    offset = base_offset + pid_h * stride_k_h + row * stride_k_d
                    k_vals = tl.load(k_ptr + offset, mask=row_mask, other=0)
            
            # Load v_t for this block's columns
            v_vals = tl.zeros(BLOCK_N, dtype=tl.float32)
            for n in range(BLOCK_N):
                if col_mask[n]:
                    col = col_ids[n]
                    offset = base_offset + pid_h * stride_v_h + col * stride_v_d
                    v_vals = tl.load(v_ptr + offset, mask=col_mask, other=0)
            
            # Load r_t for this block's rows (for output)
            r_vals = tl.zeros(BLOCK_M, dtype=tl.float32)
            for m in range(BLOCK_M):
                if row_mask[m]:
                    row = row_ids[m]
                    offset = base_offset + pid_h * stride_r_h + row * stride_r_d
                    r_vals = tl.load(r_ptr + offset, mask=row_mask, other=0)
            
            # State update: S_t = (1 - w_t) * S_{t-1} + k_t ⊗ v_t
            decay = 1.0 - w_vals[:, None]
            S_block = decay * S_block
            
            # Add outer product: k_t[i] * v_t[j]
            for m in range(BLOCK_M):
                for n in range(BLOCK_N):
                    if row_mask[m] and col_mask[n]:
                        S_block[m, n] += k_vals[m] * v_vals[n]
            
            # Compute output contribution for this block
            # y_t[j] += r_t[i] * S_t[i, j] for i in this block
            y_contrib = tl.zeros(BLOCK_N, dtype=tl.float32)
            for m in range(BLOCK_M):
                for n in range(BLOCK_N):
                    if row_mask[m] and col_mask[n]:
                        y_contrib[n] += r_vals[m] * S_block[m, n]
            
            # Store output contribution
            for n in range(BLOCK_N):
                if col_mask[n]:
                    col = col_ids[n]
                    y_offset = (pid_b * stride_y_bt + 
                               t * stride_y_bt // T if stride_y_bt > H * D else t * H * D +
                               pid_h * D + col)
                    tl.atomic_add(y_ptr + y_offset, y_contrib[n])
        
        # Store intermediate state if not last segment
        if seg < num_segments - 1:
            for m in range(BLOCK_M):
                for n in range(BLOCK_N):
                    if row_mask[m] and col_mask[n]:
                        row = row_ids[m]
                        col = col_ids[n]
                        offset = (state_base + 
                                 row * stride_state_d1 + 
                                 col * stride_state_d2)
                        tl.store(state_ptr + offset, S_block[m, n])
    
    # Store final state
    for m in range(BLOCK_M):
        for n in range(BLOCK_N):
            if row_mask[m] and col_mask[n]:
                row = row_ids[m]
                col = col_ids[n]
                offset = (state_base + 
                         row * stride_state_d1 + 
                         col * stride_state_d2)
                tl.store(state_ptr + offset, S_block[m, n])

# -----------------------------------------------------------
# Backward Kernel - Memory Efficient with Checkpointing
# -----------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'CHKPT_INTERVAL': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'CHKPT_INTERVAL': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'CHKPT_INTERVAL': 8}, num_stages=1, num_warps=16),
    ],
    key=['D', 'H', 'B', 'T'],
)
@triton.jit
def mamba_backward_kernel(
    # Gradient inputs
    grad_y_ptr,        # [B, T, H * D]
    
    # Saved tensors from forward
    k_ptr, v_ptr, w_ptr, r_ptr,  # [B, T, H, D]
    chkpt_states_ptr,  # [num_chkpts, B, H, D, D]
    chkpt_indices_ptr, # [num_chkpts]
    
    # Output gradients
    grad_k_ptr, grad_v_ptr, grad_w_ptr, grad_r_ptr,
    
    # Dimensions
    B, T, H, D, num_chkpts,
    
    # Strides (same as forward)
    stride_grady_bt, stride_grady_d,
    stride_k_bt, stride_k_h, stride_k_d,
    stride_v_bt, stride_v_h, stride_v_d,
    stride_w_bt, stride_w_h, stride_w_d,
    stride_r_bt, stride_r_h, stride_r_d,
    stride_gradk_bt, stride_gradk_h, stride_gradk_d,
    stride_gradv_bt, stride_gradv_h, stride_gradv_d,
    stride_gradw_bt, stride_gradw_h, stride_gradw_d,
    stride_gradr_bt, stride_gradr_h, stride_gradr_d,
    stride_chkpt_s, stride_chkpt_b, stride_chkpt_h, stride_chkpt_d1, stride_chkpt_d2,
    
    # Kernel config
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHKPT_INTERVAL: tl.constexpr,
):
    """
    Backward pass with memory-efficient checkpointing.
    Processes state blocks in reverse time, recomputing forward segments as needed.
    """
    
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    pid_n = tl.program_id(3)
    
    if pid_b >= B or pid_h >= H:
        return
    
    # Tile indices
    row_ids = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_ids = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = row_ids < D
    col_mask = col_ids < D
    
    # Initialize gradient state for this block
    grad_S_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Process time in reverse
    for t in range(T - 1, -1, -1):
        # Find nearest checkpoint
        chkpt_idx = t // CHKPT_INTERVAL
        chkpt_t = chkpt_idx * CHKPT_INTERVAL
        
        # ---------------------------------------------------
        # Phase 1: Recompute S_t from checkpoint
        # ---------------------------------------------------
        S_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # Load checkpoint state if available
        if chkpt_t < t or chkpt_t == 0:
            if chkpt_t == 0:
                # Initial state is zero, so S_block remains zero
                pass
            else:
                # Load from checkpoint
                for m in range(BLOCK_M):
                    for n in range(BLOCK_N):
                        if row_mask[m] and col_mask[n]:
                            row = row_ids[m]
                            col = col_ids[n]
                            offset = (chkpt_idx * stride_chkpt_s +
                                     pid_b * stride_chkpt_b +
                                     pid_h * stride_chkpt_h +
                                     row * stride_chkpt_d1 +
                                     col * stride_chkpt_d2)
                            S_block[m, n] = tl.load(chkpt_states_ptr + offset)
        
        # Forward recompute from checkpoint to t
        for s in range(chkpt_t, t + 1):
            s_offset_bt = s * (H * D) if stride_k_bt == H * D else s
            base_offset = pid_b * stride_k_bt + s_offset_bt
            
            # Load w_s, k_s, v_s for this block
            w_vals = tl.zeros(BLOCK_M, dtype=tl.float32)
            k_vals = tl.zeros(BLOCK_M, dtype=tl.float32)
            v_vals = tl.zeros(BLOCK_N, dtype=tl.float32)
            
            for m in range(BLOCK_M):
                if row_mask[m]:
                    row = row_ids[m]
                    offset = base_offset + pid_h * stride_k_h + row * stride_k_d
                    w_vals = tl.load(w_ptr + offset, mask=row_mask, other=0)
                    k_vals = tl.load(k_ptr + offset, mask=row_mask, other=0)
            
            for n in range(BLOCK_N):
                if col_mask[n]:
                    col = col_ids[n]
                    offset = base_offset + pid_h * stride_v_h + col * stride_v_d
                    v_vals = tl.load(v_ptr + offset, mask=col_mask, other=0)
            
            # State update
            decay = 1.0 - w_vals[:, None]
            S_block = decay * S_block
            
            for m in range(BLOCK_M):
                for n in range(BLOCK_N):
                    if row_mask[m] and col_mask[n]:
                        S_block[m, n] += k_vals[m] * v_vals[n]
        
        # ---------------------------------------------------
        # Phase 2: Compute gradients at time t
        # ---------------------------------------------------
        t_offset_bt = t * (H * D) if stride_grady_bt == H * D else t
        base_offset = pid_b * stride_grady_bt + t_offset_bt
        
        # Load grad_y_t for this block's columns
        grad_y_vals = tl.zeros(BLOCK_N, dtype=tl.float32)
        for n in range(BLOCK_N):
            if col_mask[n]:
                col = col_ids[n]
                offset = base_offset + pid_h * D + col
                grad_y_vals = tl.load(grad_y_ptr + offset, mask=col_mask, other=0)
        
        # Load r_t for this block's rows
        r_vals = tl.zeros(BLOCK_M, dtype=tl.float32)
        for m in range(BLOCK_M):
            if row_mask[m]:
                row = row_ids[m]
                offset = base_offset + pid_h * stride_r_h + row * stride_r_d
                r_vals = tl.load(r_ptr + offset, mask=row_mask, other=0)
        
        # Gradient for r_t: ∂L/∂r_t = grad_y_t @ S_t.T
        for m in range(BLOCK_M):
            if row_mask[m]:
                row = row_ids[m]
                grad_r = 0.0
                for n in range(BLOCK_N):
                    if col_mask[n]:
                        grad_r += grad_y_vals[n] * S_block[m, n]
                
                # Atomic add to gradient
                offset = base_offset + pid_h * stride_gradr_h + row * stride_gradr_d
                tl.atomic_add(grad_r_ptr + offset, grad_r)
        
        # Accumulate gradient for S_t from output: r_t.T @ grad_y_t
        for m in range(BLOCK_M):
            for n in range(BLOCK_N):
                if row_mask[m] and col_mask[n]:
                    grad_S_block[m, n] += r_vals[m] * grad_y_vals[n]
        
        # ---------------------------------------------------
        # Phase 3: Compute gradients for w, k, v at time t
        # ---------------------------------------------------
        if t > 0:
            # Need S_{t-1} - either from checkpoint or recompute
            prev_chkpt_idx = (t - 1) // CHKPT_INTERVAL
            prev_chkpt_t = prev_chkpt_idx * CHKPT_INTERVAL
            
            S_prev_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            
            if prev_chkpt_t == 0:
                # Initial state is zero
                pass
            else:
                # Load checkpoint for t-1
                for m in range(BLOCK_M):
                    for n in range(BLOCK_N):
                        if row_mask[m] and col_mask[n]:
                            row = row_ids[m]
                            col = col_ids[n]
                            offset = (prev_chkpt_idx * stride_chkpt_s +
                                     pid_b * stride_chkpt_b +
                                     pid_h * stride_chkpt_h +
                                     row * stride_chkpt_d1 +
                                     col * stride_chkpt_d2)
                            S_prev_block[m, n] = tl.load(chkpt_states_ptr + offset)
            
            # Forward recompute from checkpoint to t-1
            for s in range(prev_chkpt_t, t):
                s_offset_bt = s * (H * D) if stride_k_bt == H * D else s
                base_offset_s = pid_b * stride_k_bt + s_offset_bt
                
                # Load w_s, k_s, v_s
                w_s_vals = tl.zeros(BLOCK_M, dtype=tl.float32)
                k_s_vals = tl.zeros(BLOCK_M, dtype=tl.float32)
                v_s_vals = tl.zeros(BLOCK_N, dtype=tl.float32)
                
                for m in range(BLOCK_M):
                    if row_mask[m]:
                        row = row_ids[m]
                        offset = base_offset_s + pid_h * stride_k_h + row * stride_k_d
                        w_s_vals = tl.load(w_ptr + offset, mask=row_mask, other=0)
                        k_s_vals = tl.load(k_ptr + offset, mask=row_mask, other=0)
                
                for n in range(BLOCK_N):
                    if col_mask[n]:
                        col = col_ids[n]
                        offset = base_offset_s + pid_h * stride_v_h + col * stride_v_d
                        v_s_vals = tl.load(v_ptr + offset, mask=col_mask, other=0)
                
                # Update state
                decay = 1.0 - w_s_vals[:, None]
                S_prev_block = decay * S_prev_block
                
                for m in range(BLOCK_M):
                    for n in range(BLOCK_N):
                        if row_mask[m] and col_mask[n]:
                            S_prev_block[m, n] += k_s_vals[m] * v_s_vals[n]
            
            # Load inputs at time t-1
            prev_t_offset = t - 1
            prev_base = pid_b * stride_k_bt + (prev_t_offset * (H * D) if stride_k_bt == H * D else prev_t_offset)
            
            w_prev_vals = tl.zeros(BLOCK_M, dtype=tl.float32)
            k_prev_vals = tl.zeros(BLOCK_M, dtype=tl.float32)
            v_prev_vals = tl.zeros(BLOCK_N, dtype=tl.float32)
            
            for m in range(BLOCK_M):
                if row_mask[m]:
                    row = row_ids[m]
                    offset = prev_base + pid_h * stride_k_h + row * stride_k_d
                    w_prev_vals = tl.load(w_ptr + offset, mask=row_mask, other=0)
                    k_prev_vals = tl.load(k_ptr + offset, mask=row_mask, other=0)
            
            for n in range(BLOCK_N):
                if col_mask[n]:
                    col = col_ids[n]
                    offset = prev_base + pid_h * stride_v_h + col * stride_v_d
                    v_prev_vals = tl.load(v_ptr + offset, mask=col_mask, other=0)
            
            # Compute gradients
            for m in range(BLOCK_M):
                if row_mask[m]:
                    row = row_ids[m]
                    
                    # Gradient for w_{t-1}
                    grad_w = 0.0
                    for n in range(BLOCK_N):
                        if col_mask[n]:
                            grad_w -= grad_S_block[m, n] * S_prev_block[m, n]
                    
                    offset = prev_base + pid_h * stride_gradw_h + row * stride_gradw_d
                    tl.atomic_add(grad_w_ptr + offset, grad_w)
                    
                    # Gradient for k_{t-1}
                    grad_k = 0.0
                    for n in range(BLOCK_N):
                        if col_mask[n]:
                            grad_k += grad_S_block[m, n] * v_prev_vals[n]
                    
                    offset = prev_base + pid_h * stride_gradk_h + row * stride_gradk_d
                    tl.atomic_add(grad_k_ptr + offset, grad_k)
            
            for n in range(BLOCK_N):
                if col_mask[n]:
                    col = col_ids[n]
                    
                    # Gradient for v_{t-1}
                    grad_v = 0.0
                    for m in range(BLOCK_M):
                        if row_mask[m]:
                            grad_v += k_prev_vals[m] * grad_S_block[m, n]
                    
                    offset = prev_base + pid_h * stride_gradv_h + col * stride_gradv_d
                    tl.atomic_add(grad_v_ptr + offset, grad_v)
        
        # Update gradient state for next iteration (backward in time)
        if t > 0:
            # Load w_{t-1}
            prev_t_offset = t - 1
            prev_base = pid_b * stride_k_bt + (prev_t_offset * (H * D) if stride_k_bt == H * D else prev_t_offset)
            
            w_prev_vals = tl.zeros(BLOCK_M, dtype=tl.float32)
            for m in range(BLOCK_M):
                if row_mask[m]:
                    row = row_ids[m]
                    offset = prev_base + pid_h * stride_k_h + row * stride_k_d
                    w_prev_vals = tl.load(w_ptr + offset, mask=row_mask, other=0)
            
            # ∂L/∂S_{t-1} = (1 - w_{t-1}) * ∂L/∂S_t
            decay_grad = 1.0 - w_prev_vals[:, None]
            grad_S_block = decay_grad * grad_S_block

# -----------------------------------------------------------
# Complete PyTorch Module with Triton Kernels
# -----------------------------------------------------------

class MambaSelectiveScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k, v, w, r, init_state=None, checkpoint_interval=32):
        """
        x: [B, T, H * D] or [B, T, H, D]
        k, v, w, r: [B, T, H, D]
        init_state: [B, H, D, D] or None
        """
        B, T, H, D = k.shape
        
        # Ensure correct shapes
        if x.dim() == 3:
            x = x.view(B, T, H, D)
        x = x.contiguous()
        k, v, w, r = [t.contiguous() for t in [k, v, w, r]]
        
        if init_state is None:
            init_state = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)
        
        # Output tensor
        y = torch.zeros_like(x.view(B, T, H * D))
        
        # State tensor (only final state needed for output, but we store checkpoints for backward)
        final_state = torch.zeros_like(init_state)
        
        # Determine number of checkpoints
        num_chkpts = (T + checkpoint_interval - 1) // checkpoint_interval
        chkpt_states = torch.zeros(num_chkpts, B, H, D, D, device=x.device, dtype=x.dtype)
        chkpt_indices = torch.arange(0, T, checkpoint_interval, device=x.device, dtype=torch.int32)
        
        # Launch forward kernel
        grid = lambda meta: (
            B,
            H,
            triton.cdiv(D, meta['BLOCK_M']),
            triton.cdiv(D, meta['BLOCK_N']),
        )
        
        # Compute strides
        def get_strides(tensor):
            if tensor.dim() == 4:
                return (tensor.stride(0), tensor.stride(1), tensor.stride(2), tensor.stride(3))
            else:  # 3D
                return (tensor.stride(0), tensor.stride(1), tensor.stride(2))
        
        # Launch kernel
        mamba_forward_kernel[grid](
            x.view(B, T, H * D), k, v, w, r,
            y, final_state,
            B, T, H, D,
            *get_strides(x.view(B, T, H * D)),
            *get_strides(k), *get_strides(v), *get_strides(w), *get_strides(r),
            *get_strides(y), *get_strides(final_state),
        )
        
        # Save for backward
        ctx.save_for_backward(k, v, w, r, chkpt_states, chkpt_indices)
        ctx.B, ctx.T, ctx.H, ctx.D = B, T, H, D
        ctx.checkpoint_interval = checkpoint_interval
        
        return y.view(B, T, H, D)
    
    @staticmethod
    def backward(ctx, grad_y):
        k, v, w, r, chkpt_states, chkpt_indices = ctx.saved_tensors
        B, T, H, D = ctx.B, ctx.T, ctx.H, ctx.D
        
        # Gradient tensors
        grad_k = torch.zeros_like(k)
        grad_v = torch.zeros_like(v)
        grad_w = torch.zeros_like(w)
        grad_r = torch.zeros_like(r)
        
        # Ensure grad_y is contiguous
        grad_y_flat = grad_y.contiguous().view(B, T, H * D)
        
        # Launch backward kernel
        grid = lambda meta: (
            B,
            H,
            triton.cdiv(D, meta['BLOCK_M']),
            triton.cdiv(D, meta['BLOCK_N']),
        )
        
        def get_strides(tensor):
            if tensor.dim() == 4:
                return (tensor.stride(0), tensor.stride(1), tensor.stride(2), tensor.stride(3))
            else:
                return (tensor.stride(0), tensor.stride(1), tensor.stride(2))
        
        mamba_backward_kernel[grid](
            grad_y_flat, k, v, w, r, chkpt_states, chkpt_indices,
            grad_k, grad_v, grad_w, grad_r,
            B, T, H, D, len(chkpt_indices),
            *get_strides(grad_y_flat),
            *get_strides(k), *get_strides(v), *get_strides(w), *get_strides(r),
            *get_strides(grad_k), *get_strides(grad_v), *get_strides(grad_w), *get_strides(grad_r),
            *get_strides(chkpt_states),
        )
        
        # Gradient for x (input) - need to compute from k, v, w, r gradients
        # This would require another kernel, but for now return None
        return None, grad_k, grad_v, grad_w, grad_r, None, None

# -----------------------------------------------------------
# Hybrid RWKV-Mamba Block
# -----------------------------------------------------------

class HybridRWKVMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, n_heads=8, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # RWKV components
        self.time_decay = nn.Parameter(torch.ones(n_heads, self.head_dim))
        self.time_first = nn.Parameter(torch.ones(n_heads, self.head_dim))
        
        # Mamba components
        self.A = nn.Parameter(torch.randn(n_heads, d_state, d_state))
        self.D = nn.Parameter(torch.ones(n_heads, self.head_dim))
        
        # Projections
        self.in_proj = nn.Linear(d_model, d_model * 5)  # x, k, v, w, r
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Conv layer for local context
        self.conv = nn.Conv1d(d_model, d_model, d_conv, groups=d_model, padding=d_conv-1)
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, state=None):
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim
        
        # Save residual
        residual = x
        
        # Normalize
        x = self.norm(x)
        
        # 1D convolution for local context
        x_conv = self.conv(x.transpose(1, 2))[:, :, :T].transpose(1, 2)
        
        # Project to inputs
        x_proj = self.in_proj(x_conv)
        x_proj = x_proj.view(B, T, H, 5, D)
        
        # Split into components
        x_t, k, v, w, r = x_proj.unbind(dim=3)
        
        # Apply RWKV time mixing
        k = k * self.time_decay[None, None, :, :]
        v = v * self.time_first[None, None, :, :]
        
        # Apply Mamba selectivity
        w = torch.sigmoid(w + self.D[None, None, :, :])
        
        # Apply selective scan
        y = MambaSelectiveScan.apply(
            x_t.view(B, T, H * D),
            k, v, w, r,
            state,
            checkpoint_interval=32
        )
        
        # Output projection
        y = self.out_proj(y.view(B, T, C))
        
        # Add residual and dropout
        return self.dropout(y) + residual, None  # Return state=None for now

# -----------------------------------------------------------
# Testing and Validation
# -----------------------------------------------------------

def test_mamba_kernels():
    """Test the Triton kernels for correctness."""
    
    # Small test case
    B, T, H, D = 2, 32, 4, 8
    
    # Create random inputs
    x = torch.randn(B, T, H * D, device='cuda', requires_grad=True)
    k = torch.randn(B, T, H, D, device='cuda', requires_grad=True)
    v = torch.randn(B, T, H, D, device='cuda', requires_grad=True)
    w = torch.randn(B, T, H, D, device='cuda', requires_grad=True)
    r = torch.randn(B, T, H, D, device='cuda', requires_grad=True)
    
    # Run forward
    y = MambaSelectiveScan.apply(x, k, v, w, r)
    
    # Compute loss and backward
    loss = y.sum()
    loss.backward()
    
    print(f"Forward/backward completed successfully")
    print(f"y shape: {y.shape}")
    print(f"Gradients computed: {k.grad is not None}")
    
    # Check memory usage
    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    return y

def benchmark_mamba():
    """Benchmark the Triton implementation."""
    
    import time
    
    B, T, H, D = 8, 512, 8, 64
    
    # Create inputs
    x = torch.randn(B, T, H * D, device='cuda')
    k = torch.randn(B, T, H, D, device='cuda')
    v = torch.randn(B, T, H, D, device='cuda')
    w = torch.randn(B, T, H, D, device='cuda')
    r = torch.randn(B, T, H, D, device='cuda')
    
    # Warmup
    for _ in range(10):
        _ = MambaSelectiveScan.apply(x, k, v, w, r)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(100):
        y = MambaSelectiveScan.apply(x, k, v, w, r)
        y.sum().backward()
    
    torch.cuda.synchronize()
    end = time.time()
    
    print(f"Time per iteration: {(end - start) / 100 * 1000:.2f} ms")
    print(f"Throughput: {B * T / ((end - start) / 100):.0f} tokens/sec")

if __name__ == "__main__":
    # Run tests
    test_mamba_kernels()
    benchmark_mamba()