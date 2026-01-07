def unfreeze_brain(model, method='soft_reset'):
    """
    When the model's states get stuck in a bad local minimum,
    this gives it a gentle nudge.
    """
    if method == 'soft_reset':
        # Add small noise to states
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'state' in name or 'memory' in name:
                    noise = torch.randn_like(param) * 0.01 * param.std()
                    param.add_(noise)
                    param.clamp_(-1, 1)  # Keep in reasonable range
        print("🧠 Applied soft reset (added small noise to states)")
    
    elif method == 'selective_reinit':
        # Reinitialize the worst-performing components
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'ssm' in name and 'A' in name:  # State transition matrix
                    # Check if it's stuck
                    if param.grad is not None and param.grad.abs().mean() < 1e-8:
                        nn.init.normal_(param, mean=0, std=0.02)
                        print(f"🔄 Reinitialized stuck parameter: {name}")
    
    elif method == 'gradient_injection':
        # Inject synthetic gradients in dead directions
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None and param.grad.abs().max() < 1e-6:
                    # Parameter is dead, give it a kick
                    synthetic_grad = torch.randn_like(param) * 0.001
                    param.grad.add_(synthetic_grad)
                    print(f"⚡ Injected synthetic gradient to: {name}")