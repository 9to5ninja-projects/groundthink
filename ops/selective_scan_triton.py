import torch
import triton
import triton.language as tl

# =============================================================================
# TRITON KERNEL: Selective Scan (Forward)
# =============================================================================
# Logic:
#   S_t = (1 - w_t) * S_{t-1} + (k_t @ v_t.T)
#   y_t = S_t @ r_t.T  (or r_t @ S_t, depending on memory layout)
#
# Dimensions:
#   Batch (B), Heads (H), Time (T), Head_Dim (D)
#   Grid: (B * H) -> Each block processes one independent stream
#   State: [D, D] kept in SRAM (Registers/Shared Mem)
# =============================================================================

@triton.jit
def selective_scan_fwd_kernel(
    # Pointers
    k_ptr, v_ptr, w_ptr, r_ptr,
    state_ptr,     # Initial State [B, H, D, D]
    u_ptr,         # Input (Original)
    out_ptr,       # Output
    # Strides (Assuming B, H, T, D layout for inputs which is common)
    stride_b, stride_h, stride_t, stride_d,
    stride_state_b, stride_state_h, stride_state_d1, stride_state_d2,
    # Dimensions
    T, D,
    # Constants
    BLOCK_SIZE: tl.constexpr
):
    # 1. Parallelize over Batch and Heads
    pid = tl.program_id(0)
    # Reconstruct batch and head indices from linear pid
    # We passed grid=(B * H, 1, 1)
    # But usually easier to use multidim grid if supported or compute offsets
    
    # Let's assume input layout is [B, T, H, D] or [B, H, T, D]
    # The python wrapper handles the strides.
    
    # Compute pointers for this specific sequence (Batch b, Head h)
    # The offsets are calculated by the caller (Python side) or passed as params
    # Simplification: The pointers passed are already shifted to the start of the B,H sequence?
    # No, usually we pass base ptr and compute offset.
    
    # To keep arguments simpler for the v1 kernel, let's assume strides handle the jump
    # But we need 'b' and 'h' to process 'state_ptr' correctly.
    # For now, let's assume the grid is 1D array of size (B*H).
    
    off_b_h = pid # Linear index of the sequence (batch_idx * num_heads + head_idx)
    
    # Strides for K, V, W, R usually: [B, H, T, D] or [B, T, H, D]
    # We use the generic strides passed in.
    # Base offset for this B,H sequence:
    # We need to know 'b' and 'h' separately if strides are not linear...
    # BUT, if we assume fully packed [B*H, T, D], it is simpler.
    # Let's write the wrapper to reshape inputs to [B*H, T, D].
    
    # Offset into input tensors
    batch_head_offset = off_b_h * stride_h # This assumes stride_h covers stride_b logic if flattened
    # Wait, simple math: ptr + b * stride_b + h * stride_h
    # If we treat B*H as a single dimension "BatchHeads", stride is stride_h? Not necessarily.
    
    # BETTER: Just use simple pointer arithmetic for the "BatchHeads" flattened view
    # Input: [BatchHeads, T, D]
    
    # State pointer: [BatchHeads, D, D]
    state_start_ptr = state_ptr + off_b_h * (D * D) # Packed state
    
    # Output Input pointers shifted to the start of this time sequence
    k_base = k_ptr + off_b_h * stride_b # stride_b here is effectively stride for BatchHeads dimension
    v_base = v_ptr + off_b_h * stride_b 
    w_base = w_ptr + off_b_h * stride_b
    r_base = r_ptr + off_b_h * stride_b
    out_base = out_ptr + off_b_h * stride_b

    # Initialize State in Register File (SRAM)
    # D is small (64). D*D = 4096 elements.
    # 4096 * 4 bytes (fp32) = 16KB. Easily fits in SRAM.
    
    # Range for D dimension
    d_range = tl.arange(0, BLOCK_SIZE) # BLOCK_SIZE should replace D
    
    # Load State [D, D]
    # We use 2D block logic: (d_row, d_col)
    # To load effectively, we might need a separate load kernel or straightforward naive load
    # Triton JIT unrolls loops, so a double loop loading might be okay for small D
    
    # Create block pointers for the state matrix S[D, D]
    # accumulator S
    S = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32)
    
    # Load initial state if needed (skipped for v1, assume zero or load)
    # for i in range(BLOCK_SIZE):
    #     for j in range(BLOCK_SIZE):
    #         S[i, j] = tl.load(state_start_ptr + i*D + j)
    
    # TIME LOOP
    for t in range(T):
        # 1. Load Vectors k, v, w, r for time t
        # Shape [D]
        # Offset: base + t * stride_t
        t_off = t * stride_t
        
        # Load mask usually needed but T is exact here
        k = tl.load(k_base + t_off + d_range * stride_d)
        v = tl.load(v_base + t_off + d_range * stride_d)
        w = tl.load(w_base + t_off + d_range * stride_d)
        r = tl.load(r_base + t_off + d_range * stride_d)
        
        # Cast to FP32 for accumulation
        k = k.to(tl.float32)
        v = v.to(tl.float32)
        w = w.to(tl.float32)
        r = r.to(tl.float32)
        
        # 2. Update State
        # S_new = (1 - w) * S + (k^T @ v) ? No, k @ v.T [D, 1] @ [1, D] -> [D,D]
        # In Triton: broadcasting vectors to create matrix
        
        # kv_matrix = k[:, None] * v[None, :]  -> [D, D]
        kv_op = k[:, None] * v[None, :]
        
        # Decay: w is [D, 1]?
        # "state = (1 - w_chunk[:, t]) * state"
        # w is loaded as [D]. Broadcast to columns.
        decay = 1.0 - w
        decay_op = decay[:, None] # Broadcast across cols (rows get decayed)
        
        S = S * decay_op + kv_op
        
        # 3. Compute Output
        # out = r @ state
        # r is [1, D]. S is [D, D]. Result [1, D].
        # In Triton: sum(r[None, :] * S, axis=0)? NO.
        # r[1, D] @ S[D, D] -> sum over dim 0 of (r.T * S)?
        # r vector [D].
        # We need dim=0 contraction.
        # r broadcasted down rows: r_mat = r[:, None] ? No r[None, :] matches columns
        # We want: y_j = sum_i (r_i * S_ij) ??
        # Or y_i = sum_j (r_j * S_ji)?
        # Logic: ctx = r @ state.
        # r is row vector [1, D]. state is [D, D].
        # Multiply r[0, i] * S[i, j]. Sum over i.
        
        # r[:, None] -> [D, 1]. S -> [D, D]
        # This multiplies every column j by r.
        # Temp = r[:, None] * S
        # Out = sum(Temp, axis=0) -> [D]
        
        term = r[:, None] * S
        out_vec = tl.sum(term, axis=0)
        
        # 4. Store Output
        tl.store(out_base + t_off + d_range * stride_d, out_vec)

    # 5. Store Final State
    # (Omitted for brevity in initial draft, similar to load)
    
# Wrapper to call the kernel
def selective_scan_triton_forward(k, v, w, r, state=None):
    # k, v, w, r: [B, T, H, D]
    B, T, H, D = k.shape
    
    # We need the data to be contiguous for each (b, h) sequence over time.
    # Input is [B, T, H, D].
    # Transpose to [B, H, T, D] and make contiguous.
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    w = w.transpose(1, 2).contiguous()
    r = r.transpose(1, 2).contiguous()
    
    # Now shape is [B, H, T, D].
    # We can view this as [B*H, T, D].
    L = B * H
    
    # New strides for the kernel (which treats it as [L, T, D] effectively)
    stride_b = T * D  # Stride to jump to next sequence
    stride_t = D      # Stride to jump to next timestep
    stride_d = 1      # Stride to jump to next channel
    
    if state is None:
        state = torch.zeros(B, H, D, D, device=k.device, dtype=torch.float32)
    else:
        state = state.contiguous()
        
    # Output buffer [B, H, T, D]
    out = torch.empty((B, H, T, D), device=k.device, dtype=k.dtype)
    
    # Grid: One program per sequence (L)
    grid = (L, 1, 1)
    
    selective_scan_fwd_kernel[grid](
        k, v, w, r,
        state, 
        None, # u_ptr placeholder
        out,
        stride_b, 0, stride_t, stride_d, # stride_h unused
        0, 0, 0, 0, # state strides unused (assumed packed)
        T, D,
        BLOCK_SIZE=64
    )
    
    # Transpose output back to [B, T, H, D]
    return out.transpose(1, 2).contiguous()
