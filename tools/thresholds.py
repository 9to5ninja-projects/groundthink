"""
Unified Thresholds Configuration (Task 56)

Single source of truth for all pass/warn/fail thresholds across the codebase.
Import from here instead of hardcoding values.

Usage:
    from tools.thresholds import THRESHOLDS, check_status
    
    # Check a metric
    status = check_status('d1_norm_ratio', 2.5)  # Returns 'WARN'
    
    # Get raw threshold
    fail_val = THRESHOLDS['d1_norm_ratio']['fail']  # 10.0

Sources:
    - VALIDATION_ROADMAP.md: Diagnostic thresholds
    - V5_GATING.md: Scaling/comparison thresholds
    - V4_TESTING.md: Graduation gate thresholds

Reference: Task 56
Created: 2026-01-11
"""

from typing import Dict, Literal, Optional, Union
from dataclasses import dataclass


# =============================================================================
# Threshold Definitions
# =============================================================================

@dataclass
class Threshold:
    """A threshold with pass/warn/fail levels."""
    pass_max: float      # <= this = PASS
    warn_max: float      # <= this = WARN (if > pass_max)
    fail_min: float      # >= this = FAIL (explicit failure)
    direction: str       # 'lower_better' or 'higher_better'
    source: str          # Documentation source
    description: str
    
    def check(self, value: float) -> str:
        """Return 'PASS', 'WARN', or 'FAIL' based on value."""
        if self.direction == 'lower_better':
            if value >= self.fail_min:
                return 'FAIL'
            elif value > self.pass_max:
                return 'WARN'
            else:
                return 'PASS'
        else:  # higher_better
            if value <= self.fail_min:
                return 'FAIL'
            elif value < self.pass_max:
                return 'WARN'
            else:
                return 'PASS'


# =============================================================================
# All Thresholds
# =============================================================================

THRESHOLDS: Dict[str, Threshold] = {
    
    # -------------------------------------------------------------------------
    # D1-D4 Diagnostics (from VALIDATION_ROADMAP.md)
    # -------------------------------------------------------------------------
    
    'd1_norm_ratio': Threshold(
        pass_max=2.0,
        warn_max=10.0,
        fail_min=10.0,
        direction='lower_better',
        source='VALIDATION_ROADMAP.md',
        description='State norm ratio (end/start) over 512 tokens'
    ),
    
    'd2_variance_min': Threshold(
        pass_max=1e-6,      # Below this = frozen (FAIL)
        warn_max=1e-5,
        fail_min=0.0,       # Not used (we check minimum)
        direction='higher_better',
        source='VALIDATION_ROADMAP.md',
        description='Minimum state variance (below = frozen)'
    ),
    
    'd3_interaction_ratio': Threshold(
        pass_max=5.0,       # Ratio < 5x = balanced
        warn_max=10.0,
        fail_min=10.0,
        direction='lower_better',
        source='VALIDATION_ROADMAP.md',
        description='Component contribution ratio (should be balanced)'
    ),
    
    'd4_ppl_ratio': Threshold(
        pass_max=2.0,       # PPL ratio < 2x = smooth degradation
        warn_max=5.0,
        fail_min=5.0,
        direction='lower_better',
        source='VALIDATION_ROADMAP.md',
        description='Perplexity ratio at long seq vs short'
    ),
    
    # -------------------------------------------------------------------------
    # S0-S4 State Tests (from test_tiny_graduation.py)
    # -------------------------------------------------------------------------
    
    's4_state_ratio': Threshold(
        pass_max=3.0,
        warn_max=10.0,
        fail_min=10.0,
        direction='lower_better',
        source='test_tiny_graduation.py',
        description='RWKV/Mamba state norm ratio'
    ),
    
    # -------------------------------------------------------------------------
    # G1-G4 Gates (from V4_TESTING.md)
    # -------------------------------------------------------------------------
    
    'g2_entropy_min': Threshold(
        pass_max=5.0,       # 2.0-5.0 = PASS
        warn_max=7.0,       # 5.0-7.0 = WARN
        fail_min=2.0,       # Below 2.0 = collapsed
        direction='higher_better',
        source='V4_TESTING.md',
        description='Initial output entropy (diversity check)'
    ),
    
    'g4_gradient_ratio': Threshold(
        pass_max=3.0,       # 0.3-3.0 = PASS
        warn_max=10.0,      # 0.1-10 = WARN
        fail_min=10.0,
        direction='lower_better',
        source='V4_TESTING.md',
        description='RWKV/Mamba gradient ratio (balance check)'
    ),
    
    # -------------------------------------------------------------------------
    # V5 Gating - Model Comparison (from V5_GATING.md)
    # -------------------------------------------------------------------------
    
    'v5_loss_ratio': Threshold(
        pass_max=1.05,      # Within 5% = equivalent (PASS)
        warn_max=1.20,      # Within 20% = acceptable (WARN)
        fail_min=1.30,      # >30% worse = FAIL
        direction='lower_better',
        source='V5_GATING.md',
        description='Our loss / GPT-2 loss (lower = better)'
    ),
    
    'v5_loss_excellent': Threshold(
        pass_max=0.95,      # We're 5%+ better
        warn_max=1.00,
        fail_min=1.05,
        direction='lower_better',
        source='V5_GATING.md',
        description='Excellent: our loss < 0.95 * GPT-2 loss'
    ),
    
    'v5_statefulness_ratio': Threshold(
        pass_max=0.0,       # Not used (higher = better)
        warn_max=1.10,
        fail_min=1.0,       # Must be > 1.0 (better than GPT-2)
        direction='higher_better',
        source='V5_GATING.md',
        description='Our statefulness / GPT-2 statefulness'
    ),
    
    'v5_speed_ratio': Threshold(
        pass_max=0.0,       # Not used
        warn_max=1.20,      # 20% faster = WARN
        fail_min=1.50,      # 50% faster = promising
        direction='higher_better',
        source='V5_GATING.md',
        description='GPT-2 time / our time (higher = we are faster)'
    ),
    
    # -------------------------------------------------------------------------
    # Information Flow (from VALIDATION_ROADMAP.md - Task 55)
    # -------------------------------------------------------------------------
    
    'info_flow_active': Threshold(
        pass_max=0.0,       # Not used
        warn_max=0.20,
        fail_min=0.05,      # Below 5% = dead component
        direction='higher_better',
        source='VALIDATION_ROADMAP.md',
        description='Component contribution fraction (each should be >20%)'
    ),
    
    # -------------------------------------------------------------------------
    # Training Stability
    # -------------------------------------------------------------------------
    
    'loss_spike': Threshold(
        pass_max=1.5,       # <1.5x average = stable
        warn_max=2.0,
        fail_min=2.0,       # >2x = unstable
        direction='lower_better',
        source='V5_GATING.md',
        description='Current loss / moving average loss'
    ),
    
    # -------------------------------------------------------------------------
    # Long-Context Degradation (Task 60)
    # -------------------------------------------------------------------------
    
    'lc_degradation_64_512': Threshold(
        pass_max=1.5,       # <50% PPL increase = PASS
        warn_max=2.5,
        fail_min=3.0,       # >3x PPL increase = FAIL
        direction='lower_better',
        source='VALIDATION_ROADMAP.md',
        description='PPL ratio: seq512 / seq64'
    ),
}


# =============================================================================
# Helper Functions
# =============================================================================

def check_status(metric_name: str, value: float) -> str:
    """
    Check status for a metric value.
    
    Args:
        metric_name: Key in THRESHOLDS dict
        value: Observed value
        
    Returns:
        'PASS', 'WARN', or 'FAIL'
    """
    if metric_name not in THRESHOLDS:
        raise KeyError(f"Unknown metric: {metric_name}. Available: {list(THRESHOLDS.keys())}")
    return THRESHOLDS[metric_name].check(value)


def get_threshold(metric_name: str) -> Threshold:
    """Get threshold object for a metric."""
    if metric_name not in THRESHOLDS:
        raise KeyError(f"Unknown metric: {metric_name}")
    return THRESHOLDS[metric_name]


def print_all_thresholds():
    """Print all thresholds in a readable table."""
    print("\n" + "="*80)
    print(" Unified Thresholds (Task 56)")
    print("="*80)
    
    current_section = None
    for name, t in THRESHOLDS.items():
        # Detect section from name prefix
        section = name.split('_')[0].upper()
        if section != current_section:
            current_section = section
            print(f"\n[{section}]")
        
        status_example = "lower=better" if t.direction == 'lower_better' else "higher=better"
        print(f"  {name:30s} PASS≤{t.pass_max:<6g} WARN≤{t.warn_max:<6g} FAIL≥{t.fail_min:<6g} ({status_example})")
    
    print("\n" + "="*80)


def summary() -> Dict[str, Dict]:
    """Return all thresholds as a dictionary for serialization."""
    result = {}
    for name, t in THRESHOLDS.items():
        result[name] = {
            'pass_max': t.pass_max,
            'warn_max': t.warn_max,
            'fail_min': t.fail_min,
            'direction': t.direction,
            'source': t.source,
            'description': t.description,
        }
    return result


if __name__ == '__main__':
    print_all_thresholds()
    
    # Quick tests
    print("\n[Quick Tests]")
    print(f"  d1_norm_ratio=1.5 → {check_status('d1_norm_ratio', 1.5)}")
    print(f"  d1_norm_ratio=3.0 → {check_status('d1_norm_ratio', 3.0)}")
    print(f"  d1_norm_ratio=15.0 → {check_status('d1_norm_ratio', 15.0)}")
    print(f"  v5_loss_ratio=1.008 → {check_status('v5_loss_ratio', 1.008)}")
    print(f"  info_flow_active=0.003 → {check_status('info_flow_active', 0.003)}")
