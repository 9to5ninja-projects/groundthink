class GradientPathologyDetector:
    """
    SSMs develop weird gradient pathologies over time.
    This catches them early.
    """
    
    def __init__(self):
        self.gradient_history = []
        self.step_counter = 0
        
    def check(self, model):
        """Check for pathological gradients"""
        self.step_counter += 1
        
        pathology_score = 0
        warnings = []
        
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            
            grad = param.grad
            
            # Check 1: Gradient sparsity (too sparse = dying weights)
            sparsity = (grad == 0).float().mean().item()
            if sparsity > 0.9:
                pathology_score += 1
                warnings.append(f"{name}: gradient too sparse ({sparsity:.1%})")
            
            # Check 2: Gradient direction flip-flop
            if len(self.gradient_history) > 10:
                prev_grad = self.gradient_history[-1].get(name)
                if prev_grad is not None:
                    direction_change = F.cosine_similarity(
                        grad.flatten(), 
                        prev_grad.flatten(), 
                        dim=0
                    ).item()
                    if direction_change < -0.5:  # Flipped direction
                        pathology_score += 1
                        warnings.append(f"{name}: gradient direction flipped")
            
            # Check 3: Gradient magnitude collapse
            grad_norm = grad.norm().item()
            if grad_norm < 1e-7:
                pathology_score += 1
                warnings.append(f"{name}: gradient collapsed ({grad_norm:.2e})")
        
        # Store current gradients for next comparison
        current_grads = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}
        self.gradient_history.append(current_grads)
        if len(self.gradient_history) > 20:
            self.gradient_history.pop(0)
        
        # Take action based on pathology score
        if pathology_score > 5:
            print(f"🚨 CRITICAL: Gradient pathology detected (score: {pathology_score})")
            for warning in warnings[:5]:  # Show first 5 warnings
                print(f"  - {warning}")
            return True
        
        return False