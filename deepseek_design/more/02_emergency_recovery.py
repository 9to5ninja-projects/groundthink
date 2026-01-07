class EmergencyRecovery:
    """When everything goes wrong, this saves your training run"""
    
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_buffer = []
        self.max_buffer = 5  # Keep last 5 good states
        
    def save_good_state(self):
        """Save a known-good state when validation is good"""
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': getattr(self.model, 'training_step', 0)
        }
        self.checkpoint_buffer.append(state)
        if len(self.checkpoint_buffer) > self.max_buffer:
            self.checkpoint_buffer.pop(0)
    
    def recover_from_bad_state(self, loss_threshold=100.0):
        """Automatically roll back if loss explodes"""
        if len(self.checkpoint_buffer) == 0:
            return False
        
        # Load last good state
        good_state = self.checkpoint_buffer[-1]
        self.model.load_state_dict(good_state['model'])
        self.optimizer.load_state_dict(good_state['optimizer'])
        
        # Reduce learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.7
        
        print(f"🔥 EMERGENCY RECOVERY: Rolled back to step {good_state['step']}, LR reduced to {param_group['lr']}")
        return True