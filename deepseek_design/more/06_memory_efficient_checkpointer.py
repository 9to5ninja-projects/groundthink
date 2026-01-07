class MemoryEfficientCheckpointer:
    """
    Saves checkpoints without OOM killing your training.
    """
    
    def __init__(self, model, save_dir='checkpoints'):
        self.model = model
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def save_sharded(self, prefix, optimizer=None):
        """
        Save model in shards to avoid memory spikes.
        """
        print(f"💾 Saving sharded checkpoint: {prefix}")
        
        # 1. Save metadata
        metadata = {
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'dtype': str(next(self.model.parameters()).dtype),
            'timestamp': time.time()
        }
        
        with open(f'{self.save_dir}/{prefix}_metadata.json', 'w') as f:
            json.dump(metadata, f)
        
        # 2. Save model in shards
        param_shards = []
        current_shard = {}
        current_size = 0
        max_shard_size = 2 * 1024**3  # 2GB per shard
        
        for name, param in self.model.named_parameters():
            param_size = param.numel() * param.element_size()
            
            if current_size + param_size > max_shard_size and current_shard:
                # Save current shard
                shard_name = f'{prefix}_shard_{len(param_shards)}.pt'
                torch.save(current_shard, f'{self.save_dir}/{shard_name}')
                param_shards.append(shard_name)
                current_shard = {}
                current_size = 0
            
            current_shard[name] = param.cpu().detach().clone()
            current_size += param_size
        
        # Save final shard
        if current_shard:
            shard_name = f'{prefix}_shard_{len(param_shards)}.pt'
            torch.save(current_shard, f'{self.save_dir}/{shard_name}')
            param_shards.append(shard_name)
        
        # 3. Save optimizer state separately (often the largest)
        if optimizer:
            opt_shards = []
            opt_state = optimizer.state_dict()
            
            # Break up optimizer state
            for param_id, state in opt_state['state'].items():
                for key, tensor in state.items():
                    if isinstance(tensor, torch.Tensor):
                        # Save large tensors individually
                        tensor_path = f'{self.save_dir}/{prefix}_opt_{param_id}_{key}.pt'
                        torch.save(tensor.cpu(), tensor_path)
                        state[key] = tensor_path  # Replace with path
            
            # Save the rest
            opt_state_path = f'{self.save_dir}/{prefix}_optimizer.pt'
            torch.save(opt_state, opt_state_path)
        
        print(f"✅ Saved {len(param_shards)} model shards and optimizer state")