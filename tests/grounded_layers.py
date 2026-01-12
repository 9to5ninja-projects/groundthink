"""
GroundedMamba Core Layers
Per FOUNDATION.md specification
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class StableStateSSM(nn.Module):
    """
    Dynamic "thinking" with input-dependent selection.
    
    Key stability mechanisms:
    - Taylor discretization: A_d = I + A * dt (NOT matrix exponential)
    - Spectral normalization to force eigenvalues < 1
    - dt clamped to [0.001, 0.1]
    - LayerNorm after every state update
    - A init: -I * 0.1 + noise * 0.01 (ensures negative eigenvalues)
    - B, C projections: small weights (std=0.01)
    """
    
    def __init__(self, dim: int, state_dim: int = 16, dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.dt_min = dt_min
        self.dt_max = dt_max
        
        # A matrix: initialized with negative eigenvalues for stability
        # -I * 0.1 + noise * 0.01
        A_init = -torch.eye(state_dim) * 0.1 + torch.randn(state_dim, state_dim) * 0.01
        self.A = nn.Parameter(A_init)
        
        # B projection: input -> state (small init)
        self.B_proj = nn.Linear(dim, state_dim, bias=False)
        nn.init.normal_(self.B_proj.weight, std=0.01)
        
        # C projection: state -> output (small init)
        self.C_proj = nn.Linear(state_dim, dim, bias=False)
        nn.init.normal_(self.C_proj.weight, std=0.01)
        
        # D: skip connection
        self.D = nn.Parameter(torch.ones(dim))
        
        # dt projection: input-dependent time step
        self.dt_proj = nn.Linear(dim, 1, bias=True)
        # Initialize bias for moderate "thinking speed"
        self.dt_proj.bias.data.fill_(1.0)
        
        # State normalization after every update
        self.state_norm = nn.LayerNorm(state_dim)
        
    def _get_spectral_normalized_A(self):
        """Force eigenvalues < 1 via spectral normalization"""
        # Compute largest singular value
        u, s, v = torch.linalg.svd(self.A)
        max_singular = s[0]
        # Scale down if > 0.99
        if max_singular > 0.99:
            return self.A * (0.99 / max_singular)
        return self.A
    
    def forward(self, x: torch.Tensor, state: torch.Tensor = None):
        """
        Args:
            x: (B, L, dim) input
            state: (B, state_dim) previous state or None
            
        Returns:
            y: (B, L, dim) output
            new_state: (B, state_dim) updated state
        """
        B, L, D = x.shape
        
        # Initialize state if needed
        if state is None:
            state = torch.zeros(B, self.state_dim, device=x.device, dtype=x.dtype)
        
        # Get spectral-normalized A
        A = self._get_spectral_normalized_A()
        
        # Compute input-dependent dt, clamped
        dt = self.dt_proj(x)  # (B, L, 1)
        dt = torch.clamp(F.softplus(dt), self.dt_min, self.dt_max)
        dt = dt.squeeze(-1)  # (B, L)
        
        # Project input to state space
        B_x = self.B_proj(x)  # (B, L, state_dim)
        
        # Taylor discretization: A_d = I + A * dt
        I = torch.eye(self.state_dim, device=x.device, dtype=x.dtype)
        
        outputs = []
        for t in range(L):
            # Discretized A for this timestep
            A_d = I + A * dt[:, t:t+1].unsqueeze(-1)  # (B, state_dim, state_dim)
            
            # State update: state = A_d @ state + B * x
            state = torch.bmm(A_d, state.unsqueeze(-1)).squeeze(-1) + B_x[:, t]
            
            # Normalize state after EVERY update (non-negotiable)
            state = self.state_norm(state)
            
            # Output: y = C @ state + D * x
            y_t = self.C_proj(state) + self.D * x[:, t]
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)  # (B, L, dim)
        return y, state


class RWKVGroundedMemory(nn.Module):
    """
    Stable "grounding" with explicit memory bank.
    
    Per FOUNDATION.md:
    - Geometric decay across heads: decay = exp(-5.0 * (h+1) / num_heads)
    - Memory size: 1024 (explicit key-value store)
    - Time decay and time_first as learned parameters
    - alpha = 0.01 (1% blend per update)
    - wkv_state.detach() to prevent gradient explosion
    """
    
    def __init__(self, dim: int, num_heads: int = 4, memory_size: int = 1024, alpha: float = 0.01):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.memory_size = memory_size
        self.alpha = alpha  # 1% blend per update
        
        # Projections for K, V, R (receptance)
        self.key_proj = nn.Linear(dim, dim, bias=False)
        self.value_proj = nn.Linear(dim, dim, bias=False)
        self.receptance_proj = nn.Linear(dim, dim, bias=False)
        self.output_proj = nn.Linear(dim, dim, bias=False)
        
        # Time decay: learned parameter per head
        # Initialized with geometric decay: exp(-5.0 * (h+1) / num_heads)
        decay_init = torch.tensor([
            math.exp(-5.0 * (h + 1) / num_heads) 
            for h in range(num_heads)
        ])
        self.time_decay = nn.Parameter(decay_init.unsqueeze(-1).expand(-1, self.head_dim).clone())
        
        # Time first: learned parameter
        self.time_first = nn.Parameter(torch.ones(num_heads, self.head_dim) * 0.5)
        
        # Explicit memory bank (key-value store)
        # Register as buffer so it persists but doesn't get gradients
        self.register_buffer('memory_keys', torch.zeros(memory_size, dim))
        self.register_buffer('memory_values', torch.zeros(memory_size, dim))
        self.register_buffer('memory_ptr', torch.tensor(0))
        self.register_buffer('update_counter', torch.tensor(0))
        
    def forward(self, x: torch.Tensor, wkv_state: torch.Tensor = None):
        """
        Args:
            x: (B, L, dim) input
            wkv_state: (B, num_heads, head_dim, head_dim) previous state or None
            
        Returns:
            y: (B, L, dim) output
            new_state: (B, num_heads, head_dim, head_dim) updated state
        """
        B, L, D = x.shape
        H = self.num_heads
        HD = self.head_dim
        
        # Project inputs
        k = self.key_proj(x).view(B, L, H, HD)
        v = self.value_proj(x).view(B, L, H, HD)
        r = torch.sigmoid(self.receptance_proj(x)).view(B, L, H, HD)
        
        # Initialize state if needed
        if wkv_state is None:
            wkv_state = torch.zeros(B, H, HD, HD, device=x.device, dtype=x.dtype)
        
        # Detach state to prevent gradient explosion (per FOUNDATION.md)
        wkv_state = wkv_state.detach()
        
        outputs = []
        for t in range(L):
            k_t = k[:, t]  # (B, H, HD)
            v_t = v[:, t]  # (B, H, HD)
            r_t = r[:, t]  # (B, H, HD)
            
            # WKV computation with time decay
            # wkv = sum(exp(-decay * delta_t) * v_i * k_i)
            # Simplified: state update with geometric decay
            
            # Apply time decay to state
            wkv_state = wkv_state * self.time_decay.view(1, H, HD, 1)
            
            # Add new key-value outer product
            kv_outer = torch.einsum('bhd,bhe->bhde', k_t, v_t)
            wkv_state = wkv_state + kv_outer * self.time_first.view(1, H, HD, 1)
            
            # Read from state using receptance
            y_t = torch.einsum('bhde,bhd->bhe', wkv_state, r_t)
            outputs.append(y_t.reshape(B, D))
        
        y = torch.stack(outputs, dim=1)  # (B, L, D)
        y = self.output_proj(y)
        
        return y, wkv_state
    
    def update_memory(self, keys: torch.Tensor, values: torch.Tensor):
        """
        Slow memory update: alpha = 0.01, every 100 steps.
        Called externally by training loop, not every forward pass.
        
        Args:
            keys: (N, dim) keys to store
            values: (N, dim) values to store
        """
        self.update_counter += 1
        
        # Only update every 100 steps
        if self.update_counter % 100 != 0:
            return
        
        N = keys.size(0)
        ptr = self.memory_ptr.item()
        
        for i in range(N):
            idx = (ptr + i) % self.memory_size
            # Slow blend: 1% new, 99% old
            self.memory_keys[idx] = (1 - self.alpha) * self.memory_keys[idx] + self.alpha * keys[i]
            self.memory_values[idx] = (1 - self.alpha) * self.memory_values[idx] + self.alpha * values[i]
        
        self.memory_ptr = torch.tensor((ptr + N) % self.memory_size)


class GroundedMambaBlock(nn.Module):
    """
    Single block combining SSM + RWKV.
    
    Per FOUNDATION.md:
    - Fixed gate ratio during early training: SSM 0.7, RWKV 0.3
    - Learned gate exists but disabled initially
    - Gate history buffer for debugging
    """
    
    def __init__(
        self, 
        dim: int, 
        ssm_dim: int = 16, 
        num_heads: int = 4,
        fixed_gate: bool = True,  # Disable learned gate initially
        gate_buffer_size: int = 1000,
    ):
        super().__init__()
        self.dim = dim
        self.fixed_gate = fixed_gate
        
        # SSM component (dynamic "thinking")
        self.ssm = StableStateSSM(dim=dim, state_dim=ssm_dim)
        
        # RWKV component (stable "grounding")
        self.rwkv = RWKVGroundedMemory(dim=dim, num_heads=num_heads)
        
        # Learned gate (disabled initially via fixed_gate flag)
        self.gate_proj = nn.Linear(dim, 2, bias=True)
        # Initialize to approximate 0.7/0.3 split when sigmoid applied
        self.gate_proj.bias.data = torch.tensor([0.85, -0.85])  # sigmoid -> ~0.7, ~0.3
        
        # Layer norms
        self.ln_ssm = nn.LayerNorm(dim)
        self.ln_rwkv = nn.LayerNorm(dim)
        self.ln_out = nn.LayerNorm(dim)
        
        # Gate history buffer for debugging
        self.register_buffer('gate_history', torch.zeros(gate_buffer_size, 2))
        self.register_buffer('gate_ptr', torch.tensor(0))
        
    def forward(
        self, 
        x: torch.Tensor, 
        ssm_state: torch.Tensor = None,
        wkv_state: torch.Tensor = None,
    ):
        """
        Args:
            x: (B, L, dim) input
            ssm_state: previous SSM state
            wkv_state: previous RWKV state
            
        Returns:
            y: (B, L, dim) output
            states: dict with 'ssm' and 'wkv' states
        """
        B, L, D = x.shape
        
        # SSM branch
        ssm_out, ssm_state = self.ssm(self.ln_ssm(x), ssm_state)
        
        # RWKV branch
        rwkv_out, wkv_state = self.rwkv(self.ln_rwkv(x), wkv_state)
        
        # Gate computation
        if self.fixed_gate:
            # Fixed ratio: SSM 0.7, RWKV 0.3
            gate_ssm = 0.7
            gate_rwkv = 0.3
        else:
            # Learned gate
            gate_logits = self.gate_proj(x.mean(dim=1))  # (B, 2)
            gates = torch.softmax(gate_logits, dim=-1)
            gate_ssm = gates[:, 0:1].unsqueeze(1)  # (B, 1, 1)
            gate_rwkv = gates[:, 1:2].unsqueeze(1)  # (B, 1, 1)
            
            # Record gate values for debugging
            if self.training:
                ptr = self.gate_ptr.item()
                self.gate_history[ptr] = gates.mean(0).detach()
                self.gate_ptr = torch.tensor((ptr + 1) % self.gate_history.size(0))
        
        # Combine outputs
        combined = gate_ssm * ssm_out + gate_rwkv * rwkv_out
        
        # Residual + norm
        y = x + self.ln_out(combined)
        
        return y, {'ssm': ssm_state, 'wkv': wkv_state}


