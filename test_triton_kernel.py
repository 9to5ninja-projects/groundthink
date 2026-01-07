import torch
import torch.nn.functional as F
import sys
import os

# Add the current directory to path so we can import from groundthink
sys.path.append(os.path.join(os.getcwd(), 'groundthink'))

try:
    from ops.selective_scan_triton import selective_scan_triton_forward
    print("Successfully imported Triton op.")
except ImportError as e:
    print(f"Failed to import Triton op: {e}")
    sys.exit(1)

def python_reference_scan(k, v, w, r, state_init=None):
    """
    Reference implementation of Selective Scan.
    k, v, w, r: [B, T, H, D]
    """
    B, T, H, D = k.shape
    if state_init is None:
        state = torch.zeros(B, H, D, D, device=k.device, dtype=k.dtype)
    else:
        state = state_init.clone()
        
    out = torch.zeros(B, T, H, D, device=k.device, dtype=k.dtype)
    
    # Iterate over time
    for t in range(T):
        # inputs at time t: [B, H, D]
        kt = k[:, t]
        vt = v[:, t]
        wt = w[:, t]
        rt = r[:, t]
        
        # State Update: S = (1-w)S + k^T v
        # Dimensions: 
        # S: [B, H, D, D]
        # kt: [B, H, D] -> [B, H, D, 1]
        # vt: [B, H, D] -> [B, H, 1, D]
        # kv: [B, H, D, D]
        
        # Expand for bmm
        kt_u = kt.unsqueeze(-1)
        vt_u = vt.unsqueeze(-2)
        kv = torch.matmul(kt_u, vt_u) # [B, H, D, D]
        
        # Decay
        # wt: [B, H, D]. Expand to [B, H, D, 1]
        decay = (1.0 - wt).unsqueeze(-1)
        
        state = state * decay + kv
        
        # Output: y = r @ S
        # rt: [B, H, D] -> [B, H, 1, D]
        rt_u = rt.unsqueeze(-2)
        
        # out_t: [B, H, 1, D]
        out_t = torch.matmul(rt_u, state)
        
        # Squeeze back to [B, H, D]
        out[:, t] = out_t.squeeze(-2)
        
    return out

def run_test():
    torch.manual_seed(42)
    
    # Config
    B = 2
    T = 16
    H = 4
    D = 64 # Must match Triton fixed block size
    
    print(f"Testing with B={B}, T={T}, H={H}, D={D}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("Triton requires CUDA. Skipping test.")
        return

    print(f"Using device: {device}")
    
    # Generate random inputs
    k = torch.randn(B, T, H, D, device=device)
    v = torch.randn(B, T, H, D, device=device)
    w = torch.sigmoid(torch.randn(B, T, H, D, device=device)) # w in (0, 1)
    r = torch.randn(B, T, H, D, device=device)
    
    # Run Reference
    print("Running Python Reference...")
    out_ref = python_reference_scan(k, v, w, r)
    
    # Run Triton
    print("Running Triton Kernel...")
    # Triton inputs
    k_t = k.clone()
    v_t = v.clone()
    w_t = w.clone()
    r_t = r.clone()
    
    out_triton = selective_scan_triton_forward(k_t, v_t, w_t, r_t)
    
    # Check
    # We strip empty dims if any difference in shape
    if out_ref.shape != out_triton.shape:
        print(f"Shape mismatch: Ref {out_ref.shape} vs Triton {out_triton.shape}")
        
    diff = (out_ref - out_triton).abs().max().item()
    print(f"Max Difference: {diff}")
    
    if diff < 1e-3:
        print("SUCCESS: Reference and Triton outputs match!")
    else:
        print("FAILURE: Outputs verify significantly.")
        print("Reference tail:", out_ref[0, -1, 0, :5])
        print("Triton tail:   ", out_triton[0, -1, 0, :5])

if __name__ == "__main__":
    run_test()
