"""
Variance Analysis Tool for Task 0.0.1

GroundThink — Phase 0: Base Model Characterization
Copyright (c) 2026 Matthew [m_tes]

ATTRIBUTION:
Concept inspired by external analysis (Claude/researcher feedback):
    "Track output variance per layer to understand if one component 
    acts as a stabilizer while the other introduces dynamics."

OUR CONTRIBUTION:
    - VarianceTracker implementation (layer-wise statistics)
    - Stabilizer vs. destabilizer classification methodology
    - Integration with GroundThink validation framework
    - Reporting and visualization utilities

This is a novel diagnostic tool for dual-pathway model analysis.

See ATTRIBUTION.md for full citation details.

Tracks layer-wise output variance to answer:
"Is RWKV-6 a low-variance stabilizer or high-variance model?"

Usage:
    from tools.variance_analysis import VarianceTracker
    
    tracker = VarianceTracker(num_layers=8)
    
    # During training:
    for layer_idx, layer_output in enumerate(layer_outputs):
        tracker.update(layer_idx, layer_output)
    
    # Get statistics:
    stats = tracker.get_stats()
    tracker.save_report('logs/variance_analysis.txt')
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


class VarianceTracker:
    """
    Track layer-wise output variance during training.
    
    Collects mean, std, min, max for each layer's output activations
    to characterize model behavior (stabilizer vs destabilizer).
    """
    
    def __init__(self, num_layers: int):
        """
        Args:
            num_layers: Number of layers to track
        """
        self.num_layers = num_layers
        self.layer_stats = {i: [] for i in range(num_layers)}
        self.step_count = 0
    
    def update(self, layer_idx: int, output: torch.Tensor):
        """
        Record variance statistics for a layer's output.
        
        Args:
            layer_idx: Index of the layer (0-indexed)
            output: Layer output tensor [batch_size, seq_len, hidden_size]
        """
        with torch.no_grad():
            # Compute statistics across all dimensions
            flat_output = output.flatten().cpu().numpy()
            
            stats = {
                'mean': float(np.mean(flat_output)),
                'std': float(np.std(flat_output)),
                'min': float(np.min(flat_output)),
                'max': float(np.max(flat_output)),
                'var': float(np.var(flat_output)),
            }
            
            self.layer_stats[layer_idx].append(stats)
    
    def update_from_activations(self, activations: List[torch.Tensor]):
        """
        Convenience method to update all layers from activation list.
        
        Args:
            activations: List of layer outputs [num_layers × [B, L, H]]
        """
        for layer_idx, output in enumerate(activations):
            if layer_idx < self.num_layers:
                self.update(layer_idx, output)
        self.step_count += 1
    
    def get_stats(self) -> Dict[int, Dict[str, float]]:
        """
        Get aggregated statistics for all layers.
        
        Returns:
            dict: {
                layer_idx: {
                    'mean_avg': float,
                    'std_avg': float,
                    'var_avg': float,
                    'range_avg': float,
                }
            }
        """
        aggregated = {}
        
        for layer_idx, stats_list in self.layer_stats.items():
            if not stats_list:
                continue
            
            # Aggregate across all recorded steps
            aggregated[layer_idx] = {
                'mean_avg': np.mean([s['mean'] for s in stats_list]),
                'std_avg': np.mean([s['std'] for s in stats_list]),
                'var_avg': np.mean([s['var'] for s in stats_list]),
                'range_avg': np.mean([s['max'] - s['min'] for s in stats_list]),
                'min_global': np.min([s['min'] for s in stats_list]),
                'max_global': np.max([s['max'] for s in stats_list]),
            }
        
        return aggregated
    
    def get_variance_trend(self) -> Dict[str, float]:
        """
        Analyze variance trend across layers.
        
        Returns:
            dict: {
                'initial_var': float,      # Layer 0 variance
                'final_var': float,        # Last layer variance
                'var_increase': float,     # Percent increase
                'is_stabilizer': bool,     # True if variance decreases
            }
        """
        stats = self.get_stats()
        
        if not stats or 0 not in stats or (self.num_layers - 1) not in stats:
            return {}
        
        initial_var = stats[0]['var_avg']
        final_var = stats[self.num_layers - 1]['var_avg']
        
        var_change = final_var - initial_var
        var_increase_pct = (var_change / initial_var * 100) if initial_var > 0 else 0.0
        
        return {
            'initial_var': initial_var,
            'final_var': final_var,
            'var_change': var_change,
            'var_increase_pct': var_increase_pct,
            'is_stabilizer': var_change < 0,  # Variance decreases = stabilizer
        }
    
    def save_report(self, output_path: str):
        """
        Save variance analysis report to text file.
        
        Args:
            output_path: Path to save report
        """
        stats = self.get_stats()
        trend = self.get_variance_trend()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("Variance Analysis Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Total steps analyzed: {self.step_count}\n")
            f.write(f"Number of layers: {self.num_layers}\n\n")
            
            # Overall trend
            if trend:
                f.write("Overall Trend:\n")
                f.write(f"  Initial variance (Layer 0): {trend['initial_var']:.6f}\n")
                f.write(f"  Final variance (Layer {self.num_layers-1}): {trend['final_var']:.6f}\n")
                f.write(f"  Variance change: {trend['var_change']:+.6f} ({trend['var_increase_pct']:+.2f}%)\n")
                f.write(f"  Behavior: {'STABILIZER (variance decreases)' if trend['is_stabilizer'] else 'DESTABILIZER (variance increases)'}\n")
                f.write("\n")
            
            # Layer-by-layer statistics
            f.write("Layer-wise Statistics:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Layer':<8} {'Mean':<12} {'Std':<12} {'Variance':<12} {'Range':<12}\n")
            f.write("-" * 60 + "\n")
            
            for layer_idx in range(self.num_layers):
                if layer_idx in stats:
                    s = stats[layer_idx]
                    f.write(f"{layer_idx:<8} {s['mean_avg']:<12.6f} {s['std_avg']:<12.6f} "
                           f"{s['var_avg']:<12.6f} {s['range_avg']:<12.6f}\n")
            
            f.write("-" * 60 + "\n")
        
        print(f"✓ Variance report saved: {output_path}")
    
    def get_comparison_summary(self) -> str:
        """
        Get a concise summary for comparison with other models.
        
        Returns:
            str: One-line summary
        """
        trend = self.get_variance_trend()
        
        if not trend:
            return "Insufficient data for variance analysis"
        
        behavior = "stabilizer" if trend['is_stabilizer'] else "destabilizer"
        return (f"Variance: {trend['initial_var']:.6f} → {trend['final_var']:.6f} "
                f"({trend['var_increase_pct']:+.2f}%) | Behavior: {behavior.upper()}")


# Quick test
if __name__ == "__main__":
    print("Testing VarianceTracker...\n")
    
    # Simulate layer outputs
    num_layers = 8
    tracker = VarianceTracker(num_layers)
    
    # Simulate 5 training steps
    for step in range(5):
        print(f"Step {step + 1}:")
        layer_outputs = []
        
        for layer_idx in range(num_layers):
            # Simulate layer output: variance increases with depth
            base_variance = 0.1 + (layer_idx * 0.05)
            output = torch.randn(2, 64, 144) * base_variance  # [B, L, H]
            layer_outputs.append(output)
            print(f"  Layer {layer_idx}: var ≈ {base_variance:.3f}")
        
        tracker.update_from_activations(layer_outputs)
        print()
    
    # Get results
    print("Analysis Results:")
    print("-" * 40)
    trend = tracker.get_variance_trend()
    print(f"Initial variance: {trend['initial_var']:.6f}")
    print(f"Final variance: {trend['final_var']:.6f}")
    print(f"Change: {trend['var_change']:+.6f} ({trend['var_increase_pct']:+.2f}%)")
    print(f"Behavior: {'STABILIZER' if trend['is_stabilizer'] else 'DESTABILIZER'}")
    print()
    print(tracker.get_comparison_summary())
    
    # Save report
    tracker.save_report('/tmp/variance_test.txt')
