import torch
from torch.utils.checkpoint import checkpoint

# =============================================================================
# Selective Scan - Using PyTorch's Native Gradient Checkpointing
# =============================================================================
# Uses torch.utils.checkpoint for memory-efficient backward
# Let PyTorch handle the complexity
# =============================================================================

SEGMENT_SIZE = 64  # Process this many timesteps per checkpointed segment


def _segment_forward(k_seg, v_seg, w_seg, r_seg, S_init):
    """Process a segment of timesteps. This function will be checkpointed."""
    B, T_seg, H, D = k_seg.shape
    device = k_seg.device
    
    S = S_init
    outputs = []
    
    for t in range(T_seg):
        w_t = w_seg[:, t].unsqueeze(-1)  # [B, H, D, 1]
        k_t = k_seg[:, t].unsqueeze(-1)  # [B, H, D, 1]
        v_t = v_seg[:, t].unsqueeze(-2)  # [B, H, 1, D]
        r_t = r_seg[:, t].unsqueeze(-2)  # [B, H, 1, D]
        
        S = (1.0 - w_t) * S + k_t * v_t
        y_t = torch.matmul(r_t, S).squeeze(-2)
        outputs.append(y_t)
    
    return torch.stack(outputs, dim=1), S


def selective_scan_triton_forward(k, v, w, r, state=None):
    """
    Memory-efficient selective scan using PyTorch gradient checkpointing.
    
    Args:
        k, v, w, r: [B, T, H, D]
        state: optional initial state [B, H, D, D]
    
    Returns:
        output: [B, T, H, D]
    """
    B, T, H, D = k.shape
    device = k.device
    
    if state is None:
        S = torch.zeros(B, H, D, D, device=device, dtype=torch.float32)
    else:
        S = state.float()
    
    num_segments = (T + SEGMENT_SIZE - 1) // SEGMENT_SIZE
    all_outputs = []
    
    for seg_idx in range(num_segments):
        start = seg_idx * SEGMENT_SIZE
        end = min(start + SEGMENT_SIZE, T)
        
        k_seg = k[:, start:end].float()
        v_seg = v[:, start:end].float()
        w_seg = w[:, start:end].float()
        r_seg = r[:, start:end].float()
        
        # Use gradient checkpointing - PyTorch will recompute forward during backward
        seg_out, S = checkpoint(
            _segment_forward,
            k_seg, v_seg, w_seg, r_seg, S,
            use_reentrant=False
        )
        all_outputs.append(seg_out)
    
    return torch.cat(all_outputs, dim=1).to(k.dtype)


# =============================================================================
# Gradient Check
# =============================================================================
def check_gradients():
    """Compare against PyTorch autograd."""
    torch.manual_seed(42)
    B, T, H, D = 2, 128, 2, 8
    device = 'cuda'
    
    def reference_forward(k, v, w, r):
        """Reference without checkpointing."""
        S = torch.zeros(B, H, D, D, device=device)
        outputs = []
        for t in range(T):
            w_t = w[:, t].unsqueeze(-1)
            k_t = k[:, t].unsqueeze(-1)
            v_t = v[:, t].unsqueeze(-2)
            r_t = r[:, t].unsqueeze(-2)
            S = (1 - w_t) * S + k_t * v_t
            y_t = torch.matmul(r_t, S).squeeze(-2)
            outputs.append(y_t)
        return torch.stack(outputs, dim=1)
    
    # Create inputs
    k = torch.randn(B, T, H, D, device=device, requires_grad=True, dtype=torch.float32)
    v = torch.randn(B, T, H, D, device=device, requires_grad=True, dtype=torch.float32)
    w = torch.sigmoid(torch.randn(B, T, H, D, device=device)).clone().detach().requires_grad_(True)
    r = torch.randn(B, T, H, D, device=device, requires_grad=True, dtype=torch.float32)
    
    # Reference
    y_ref = reference_forward(k, v, w, r)
    y_ref.sum().backward()
    grad_k_ref = k.grad.clone()
    grad_v_ref = v.grad.clone()
    grad_w_ref = w.grad.clone()
    grad_r_ref = r.grad.clone()
    
    k.grad = v.grad = w.grad = r.grad = None
    
    # Our version
    y_ours = selective_scan_triton_forward(k, v, w, r)
    y_ours.sum().backward()
    
    print(f"Output diff: {(y_ref - y_ours).abs().max().item():.2e}")
    print(f"grad_k diff: {(grad_k_ref - k.grad).abs().max().item():.2e}")
    print(f"grad_v diff: {(grad_v_ref - v.grad).abs().max().item():.2e}")
    print(f"grad_w diff: {(grad_w_ref - w.grad).abs().max().item():.2e}")
    print(f"grad_r diff: {(grad_r_ref - r.grad).abs().max().item():.2e}")
    
    all_ok = all([
        (y_ref - y_ours).abs().max().item() < 1e-4,
        (grad_k_ref - k.grad).abs().max().item() < 1e-3,
        (grad_v_ref - v.grad).abs().max().item() < 1e-3,
        (grad_w_ref - w.grad).abs().max().item() < 1e-3,
        (grad_r_ref - r.grad).abs().max().item() < 1e-3,
    ])
    print(f"\nGradient check: {'PASSED' if all_ok else 'FAILED'}")
    return all_ok


if __name__ == "__main__":
    check_gradients()


if __name__ == "__main__":
    check_gradients()
