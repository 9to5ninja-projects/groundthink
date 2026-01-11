# V4 Strategy Archive - Completed Task Details

**Archived:** 2026-01-10  
**Source:** V4_STRATEGY.md  
**Purpose:** Historical reference for completed tasks. Do not edit.

---

## Completed Tasks (Reference Only)

<details>
<summary>Tasks 1-5: Foundation Setup (Click to expand)</summary>

### Task 1: Verify data_v030.py + tokenizer_v030.py Work
**Status:** ✅ COMPLETE (2026-01-09)  
Files now in `archive/` but still functional. train_v4.py imports from archive.

### Task 2: Create/Validate Test Dataset
**Status:** ✅ COMPLETE (2026-01-09)  
Dataset: shakespeare.txt (1.1M tokens, char-level, 97 vocab size)

### Task 3: Build First Hybrid (ParallelHybridBlock)
**Status:** ✅ COMPLETE (2026-01-09)  
Output: `hybrid_v4.py` - 3.8M parameters, RWKV6 + Mamba2 parallel architecture

### Task 4: Define Training Configuration
**Status:** ✅ COMPLETE (2026-01-09)  
Output: `train_v4.py` - Full training script with gradient monitoring, differential LR

### Task 5: First Training Run
**Status:** ✅ COMPLETE (2026-01-09)  
Results: Loss 1.37, perplexity 3.0, 33K tok/s, gradient ratio warning (0.15-0.16)

</details>

---

## Archived Task Details (Reference Only)

> **Note:** Tasks 6.5-6.12 were SUPERSEDED by Phase 0 CUDA integration. 
> These descriptions are kept for historical reference only.
> See Phase 0 and Phase 1 tables above for actual completion status.

### Task 6.6: Research RWKV-6 Architecture

**Status:** ✅ SUPERSEDED by Task 0.1  
**Complexity:** M (Medium)  
**Time:** ~1 hour  
**Scope:** Find authoritative RWKV-6 specifications and document requirements

**Decision Context:**
- FLA library not installed, custom wrappers in `archive/cuda_backends.py` are simplified/fake
- Cannot use FLA (per user decision)
- Must build correct RWKV-6 and Mamba-2 from scratch
- **Critical:** V3 failed by making up components - we must get this right

**Research Objectives:**
1. Find official RWKV-6 paper/specification
2. Identify key architectural components:
   - Time mixing mechanism (WKV kernel)
   - Channel mixing
   - Token/time shift
   - State management
3. Document mathematical formulas
4. Find reference implementations (RWKV-LM official repo)
5. Identify critical differences vs RWKV-5

**Deliverables:**
- Create `RWKV6_SPEC.md` with:
  - Paper citations
  - Architecture diagram/description
  - Mathematical formulas
  - Key parameters (num_heads, head_dim, etc.)
  - Critical implementation notes
  - Links to reference code

**Acceptance Criteria:**
- [ ] Official RWKV-6 paper/spec found and documented
- [ ] Key components identified and described
- [ ] Mathematical formulas transcribed
- [ ] Reference implementation links saved
- [ ] RWKV6_SPEC.md created in workspace

**Resources:**
- RWKV official GitHub: https://github.com/BlinkDL/RWKV-LM
- Papers: arXiv search for "RWKV-6"
- Community: RWKV Discord/discussions

---

### Task 6.7: Research Mamba-2 Architecture

**Status:** ✅ SUPERSEDED by Task 0.2  
**Complexity:** M (Medium)  
**Time:** ~1 hour  
**Scope:** Find authoritative Mamba-2 specifications and document requirements

**Research Objectives:**
1. Find official Mamba-2 paper (Dao & Gu)
2. Identify key components:
   - State-space model (SSM) formulation
   - Selective scan mechanism
   - SSD (Structured State-Space Duality)
   - Chunk-wise processing
3. Document mathematical formulas
4. Find reference implementations (state-spaces/mamba repo)
5. Hardware requirements (CUDA kernels, Triton)

**Deliverables:**
- Create `MAMBA2_SPEC.md` with:
  - Paper citations
  - Architecture description
  - SSM formulation
  - Key parameters (d_state, d_conv, expand)
  - Implementation notes
  - Links to reference code

**Acceptance Criteria:**
- [ ] Official Mamba-2 paper found and documented
- [ ] SSM mechanism understood and described
- [ ] Mathematical formulas transcribed
- [ ] Reference implementation links saved
- [ ] MAMBA2_SPEC.md created in workspace

---

### Task 6.8: Audit Custom Wrappers

**Status:** ✅ SUPERSEDED by Phase 0  
**Complexity:** M (Medium)  
**Time:** ~1-2 hours  
**Scope:** Compare cuda_backends.py against official specs

**What to Check:**

**RWKV6Attention (lines 9-62 in cuda_backends.py):**
- [ ] Time mixing uses correct WKV formula
- [ ] Channel mixing implemented correctly
- [ ] Token shift mechanism present
- [ ] State management correct
- [ ] Return signature matches: (output, attn_weights, past_kv)

**Mamba2 (lines 65-113 in cuda_backends.py):**
- [ ] SSM formulation correct
- [ ] Selective scan mechanism present
- [ ] Conv1d usage correct (d_conv parameter)
- [ ] Expansion factor correct
- [ ] State management correct

**Deliverables:**
- Create `WRAPPER_AUDIT.md` documenting:
  - What's correct
  - What's missing
  - What's wrong
  - Severity of issues (blocker/warning/minor)
  - Recommendations (fix vs rebuild)

**Acceptance Criteria:**
- [ ] Line-by-line comparison completed
- [ ] All discrepancies documented
- [ ] Severity assessed for each issue
- [ ] Clear recommendation: fix wrappers or rebuild from scratch

---

### Task 6.9: Implement/Fix RWKV-6 Component

**Status:** ✅ SUPERSEDED by Task 0.1  
**Complexity:** L/XL (Large or Extra Large - TBD after audit)  
**Time:** ~2-8 hours (depends on audit results)  
**Scope:** Build correct RWKV-6 implementation

**Will be detailed after Task 6.8 audit determines scope.**

**Acceptance Criteria:**
- [ ] Passes G1 gate (forward pass, no NaN)
- [ ] Matches RWKV-6 spec exactly
- [ ] Documented in V4_BUILD_LOG.md

---

### Task 6.10: Implement/Fix Mamba-2 Component

**Status:** ✅ SUPERSEDED by Task 0.2  
**Complexity:** L/XL (Large or Extra Large - TBD after audit)  
**Time:** ~2-8 hours (depends on audit results)  
**Scope:** Build correct Mamba-2 implementation

**Will be detailed after Task 6.8 audit determines scope.**

**Acceptance Criteria:**
- [ ] Passes G1 gate (forward pass, no NaN)
- [ ] Matches Mamba-2 spec exactly
- [ ] Documented in V4_BUILD_LOG.md

---

### Task 6.11: Rebuild hybrid_v4.py

**Status:** ✅ COMPLETE (cuda_backends.py)  
**Complexity:** L (Large)  
**Time:** ~2-3 hours  
**Scope:** Integrate verified RWKV-6 and Mamba-2 components

**What to Do:**
1. Move verified wrappers to root as `rwkv6_component.py` and `mamba2_component.py`
2. Update hybrid_v4.py imports
3. Verify ParallelHybridBlock architecture unchanged
4. Test forward pass
5. Run G1 and G2 gates

**Acceptance Criteria:**
- [ ] Imports work: `from rwkv6_component import RWKV6Attention`
- [ ] Passes G1 gate (forward pass)
- [ ] Passes G2 gate (init entropy 2.0-5.0)
- [ ] Model parameters ~3.8M (same as before)
- [ ] Documented in V4_BUILD_LOG.md

---

### Task 6.12: Verify Model Works

**Status:** ✅ COMPLETE (test_phase0_complete.py)  
**Complexity:** M (Medium)  
**Time:** ~1 hour  
**Scope:** Full validation before proceeding to optimization

**Tests:**
1. Import test: `import hybrid_v4` succeeds
2. Instantiation: Model builds without errors
3. Forward pass: Process sample batch
4. G1 gate: No NaN, correct output shapes
5. G2 gate: Init entropy in healthy range
6. Quick training: 10 steps to verify gradients flow

**Acceptance Criteria:**
- [ ] All tests pass
- [ ] Gates G1-G2 passed
- [ ] Ready for Task 6.5 (monitoring test)
- [ ] Results documented in V4_BUILD_LOG.md

---

### Task 6.5: Test Monitoring Tools

**Status:** ✅ COMPLETE  
**Complexity:** S (Small)  
**Time:** ~15-20 minutes  
**Scope:** Verify monitoring tools work during actual training

**Why This Task:**
Standard operating procedure - we installed tools but didn't verify they work together under load. Need to test before using them for profiling analysis.

**What to Do:**
1. Start nvidia-smi logging in background: `nvidia-smi -l 1 > logs/monitoring/test_run_$(date +%Y%m%d_%H%M%S).log &`
2. Run short training session: `python train_v4.py --steps 100 --no-checkpoint`
3. Monitor in real-time with: `nvtop` (in separate terminal)
4. Stop nvidia-smi logger after training completes
5. Verify log file captured GPU metrics during training

**Acceptance Criteria:**
- [ ] Training runs successfully with monitoring active
- [ ] nvidia-smi log file shows GPU utilization >0% during training
- [ ] nvtop displays real-time stats (can screenshot or describe observed metrics)
- [ ] Baseline under load documented: GPU util %, VRAM usage, temperature, power draw
- [ ] No conflicts or performance degradation from monitoring tools

**What to Document in V4_BUILD_LOG.md:**
- GPU utilization % during training (target: should see >50%)
- VRAM usage under load (expect ~422 MiB based on previous run)
- Temperature and power draw peaks
- Any anomalies or issues with monitoring tools

---

### Task 7: Baseline Performance Profiling

**Status:** ⬜ **NEXT**  
**Complexity:** M (Medium)  
**Time:** ~1 hour  
**Scope:** Create reusable benchmark suite + record baseline measurements

**Why This Matters:**
- Cannot optimize without knowing current performance
- Need fixed benchmark configs for reproducible comparisons
- Existing metrics scattered across train_v4.py + docs, need consolidation

**Existing Assets:**
- `train_v4.py`: Runtime metrics (tok/s, loss, R/M ratio, entropy, activation stats)
- `V4_DIAGNOSTICS.md`: KernelBenchmark class specification (lines 545-700)
- `V4_TESTING.md`: Benchmark snippets (speed_test, memory_test)
- `test_phase0_complete.py`: Reusable test pattern (7 tests + 5 gates)

**Sub-Tasks:**

| # | Sub-Task | Status | Output | Complexity |
|---|----------|--------|--------|------------|
| 7.1 | Create benchmark_suite.py | ⬜ | Reusable benchmark script | M |
| 7.2 | Run B1: Throughput test | ⬜ | tok/s at fixed batch/seq | S |
| 7.3 | Run B2: Memory test | ⬜ | Peak VRAM at fixed config | S |
| 7.4 | Run B3: Stability test | ⬜ | 100-step loss delta | S |
| 7.5 | Document baseline in V4_BUILD_LOG.md | ⬜ | Session 9 entry | S |

**Benchmark Definitions (Fixed Configs):**

| Benchmark | Config | Metric | Target | Pass Criteria |
|-----------|--------|--------|--------|---------------|
| B1: Throughput | batch=8, seq=64, 100 steps | tok/s | Measure baseline | Record value |
| B2: Memory | batch=8, seq=64 | Peak VRAM (MiB) | <1000 MiB | Record value |
| B3: Stability | batch=8, seq=64, 100 steps | Loss Δ | Decreasing | loss_end < loss_start |

**Acceptance Criteria:**
- [ ] benchmark_suite.py created (reusable like test_phase0_complete.py)
- [ ] All 3 benchmarks run with fixed configs
- [ ] Baseline numbers recorded in V4_BUILD_LOG.md Session 9
- [ ] Ready for Task 8 (optimizations) comparison

---

### Task 8: Apply Quick Win Optimizations

**Status:** ⬜ PENDING  
**Complexity:** L (Large)  
**Time:** ~2-3 hours  
**Scope:** Test each optimization using benchmark_suite.py from Task 7

**Dependencies:** Task 7 (baseline measurements required for comparison)

**Optimizations to test (one at a time):**

| # | Optimization | Config Change | Expected Improvement |
|---|--------------|---------------|----------------------|
| 8.1 | Larger batch | batch: 8 → 16 | ~1.5-2x tok/s |
| 8.2 | DataLoader workers | workers: 0 → 4, pin_memory=True | ~1.2x tok/s |
| 8.3 | Mixed precision (AMP) | torch.cuda.amp | ~1.5-2x tok/s |
| 8.4 | torch.compile | model = torch.compile(model) | ~1.3-2x tok/s |

**Testing Protocol:**
1. Run benchmark_suite.py with ONE change
2. Compare to Task 7 baseline (B1, B2, B3)
3. If B3 passes (loss decreasing), optimization is valid
4. Record improvement factor

**Acceptance Criteria:**
- [ ] Each optimization tested independently with benchmark_suite.py
- [ ] Comparison table with baseline vs each optimization
- [ ] No quality degradation (B3 must still pass)
- [ ] Best single optimization identified

---

### Task 10: Run Controlled Experiments

**Status:** ⬜ PENDING  
**Time:** ~3-4 hours  
**Scope:** Systematic testing of optimization combinations (V4.5_OPTIMIZATION.md Phase 4)

**Experiment matrix:**
- Baseline (8 batch, no workers, fp32, no compile)
- +Batch size increase
- +Workers
- +AMP
- +torch.compile (all optimizations)

**Acceptance Criteria:**
- [ ] Run each config for 1000 steps
- [ ] Log comparison table with throughput, VRAM, loss
- [ ] Validate no quality degradation (loss within ±0.05)
- [ ] Document speedup multiplier

---

### Task 11: Select Optimal Configuration

**Status:** ⬜ PENDING  
**Time:** ~30 minutes  
**Scope:** Choose best setup and update train_v4.py defaults

**Requirements:**
- Compare all experiment results
- Choose configuration with best throughput/quality trade-off
- Update train_v4.py CONFIG with optimal settings
- Document final configuration

**Acceptance Criteria:**
- [ ] Optimal config selected (target: 5x baseline throughput)
- [ ] train_v4.py updated with new defaults
- [ ] Results documented in V4_BUILD_LOG.md

---

### Task 11: Analyze Training Results

**Status:** ⬜ PENDING  
**Time:** ~1-2 hours  
**Scope:** Deep analysis of baseline and optimized training runs with proper monitoring tools

**Requirements:**
- Review baseline training curves (5000 steps)
- Review optimized training curves (from Task 9-10)
- Compare gradient dynamics across configurations
- Analyze component contributions (RWKV vs Mamba)
- Check for activation collapse or state issues

**What We Now Have for Analysis:**
- Performance profiles (torch.profiler traces)
- GPU utilization data (nvtop logs)
- Throughput comparisons (baseline vs optimized)
- Loss convergence curves

**Acceptance Criteria:**
- [ ] Training curves analyzed and documented
- [ ] Gradient ratio patterns understood
- [ ] Component health validated
- [ ] Recommendations for hyperparameter tuning (Task 12)
- [ ] Document findings in V4_BUILD_LOG.md

---

### Task 12: Address Gradient Imbalance

**Status:** ⬜ PENDING  
**Time:** ~1-2 hours  
**Scope:** Fix RWKV/Mamba gradient ratio (currently 0.15, target 0.3-3.0)

**Analysis needed:**
- Why is RWKV gradient 6-7x weaker than Mamba?
- Is mamba_lr_mult=2.0 too high?
- Does RWKV need better initialization?

**Potential fixes:**
- Adjust mamba_lr_mult (try 1.5, 1.0, 0.5)
- Increase RWKV learning rate independently
- Check RWKV6 initialization in hybrid_v4.py

**Acceptance Criteria:**
- [ ] Root cause identified
- [ ] Solution tested (1000 steps)
- [ ] Gradient ratio in healthy range (0.3-3.0)
- [ ] Document fix in V4_BUILD_LOG.md

---

### Task 13: Extended Training Run

**Status:** ✅ COMPLETE (2026-01-09)  
**Time:** 582.4s (~10 min) at 35K tok/s avg  
**Scope:** Train optimized model to convergence (5000 steps)

**Results:**
- Final Train Loss: 1.1375
- Final Val Loss: 1.4916
- Best Val Loss: 1.4607 (step ~4500)
- Final PPL: 3.12 train / 4.44 val
- Entropy: 3.83 → 3.91 (healthy growth)
- Gradient Ratio: Started 0.4-0.5, drifted to 0.29 at end (low LR)
- Checkpoints: 6 saved (1K, 2K, 3K, 4K, 5K, final)

**Acceptance Criteria:**
- [x] Training completes without crashes
- [x] Val loss converges or plateaus (1.46 best)
- [x] Final model checkpoint saved (ckpt_HY_final.pt)
- [x] Training curves logged and analyzed
- [x] Results documented

**Observation:** Gradient ratio drifted from 0.4-0.5 (mid-training) to 0.28-0.33 (late training) as LR decayed via cosine schedule. This is expected behavior - RWKV layers have proportionally lower gradients when LR is very small. Model convergence was excellent regardless.

---

<!-- Legacy task descriptions removed (2026-01-10) - see Phase 1 table for completion status -->

---

### Task 22: Long-Context Retrieval Test (NIAH)

**Status:** ⬜ PENDING  
**Time:** ~1-2 hours  
**Scope:** Validate model's long-context memory retention capability

**What is NIAH?**
Needle-in-a-Haystack test hides key facts ("needles") in filler text ("haystack") and queries retrieval accuracy. Reveals "context rot" where models fail beyond claimed context length.

**Why test this?**
- Hybrid RWKV/Mamba architecture claims better long-context handling than pure attention
- Need empirical validation of context window effectiveness (not just theoretical claims)
- Identifies degradation points before scaling to larger models

**Setup:**
```bash
# Install testing framework
pip install needlehaystack

# Test at multiple context lengths
needlehaystack.run_test \
  --provider custom \
  --model_name "hybrid_v4_8M" \
  --context_lengths "[1000,2000,4000,8000,16000]" \
  --document_depth_percents "[10,25,50,75,90]" \
  --multi_needle False \
  --save_results True
```

**Test protocol:**
1. **Single needle (baseline):** One fact hidden in Paul Graham essays
   - Needle: "The best pizza topping is mushrooms."
   - Query: "What is the best pizza topping?"
   - Test at 1K, 2K, 4K, 8K, 16K tokens

2. **Depth variation:** Place needle at start (10%), middle (50%), end (90%)
   - Check if middle degrades (common attention bias)

3. **Multi-needle (advanced):** 10 facts spaced evenly
   - Tests multi-fact recall and memory interference

**Custom model integration:**
```python
from needlehaystack import LLMNeedleHaystackTester

class HybridModelTester(LLMNeedleHaystackTester):
    def __init__(self, model_path):
        self.model = load_hybrid_model(model_path)
    
    def evaluate_model(self, context: str, question: str) -> str:
        prompt = context + "\n\n" + question
        output = self.model.generate(prompt, max_tokens=50)
        return output
```

**Expected Results:**
- **Target:** >80% accuracy up to claimed context length
- **Baseline comparison:** Document vs GPT-3.5 (degrades at 4K-8K typically)
- **Hybrid advantage:** Should outperform pure attention at 8K+

**Analysis:**
- Plot accuracy heatmap (context length vs needle depth)
- Identify degradation point (where accuracy drops below 80%)
- Compare RWKV-heavy vs Mamba-heavy configurations
- Document in V4_BUILD_LOG.md

**Acceptance Criteria:**
- [ ] Tests run successfully at 1K, 2K, 4K, 8K, 16K tokens
- [ ] Accuracy measured for start/middle/end positions
- [ ] Results visualized (heatmap or line plot)
- [ ] Degradation point identified
- [ ] Results compared to baseline (GPT-3.5 or similar)
- [ ] Findings documented in V4_BUILD_LOG.md

**Red Flags:**
- Accuracy <80% at 4K tokens (indicates poor long-context handling)
- Middle positions significantly worse than start/end
- Multi-needle accuracy <50% (memory interference)

**See Also:** [V4.5_OPTIMIZATION.md](V4.5_OPTIMIZATION.md) - Full NIAH methodology and setup

---

### Task 25: LongBench Evaluation (Multitask Real-World Memory)

**Status:** ⬜ PENDING  
**Time:** ~2-4 hours  
**Scope:** Evaluate holistic memory on diverse real-world tasks (QA, summarization, code)

**Prerequisites:**
- Task 22 (NIAH) must pass with >80% accuracy at 16K tokens
- Model demonstrates basic long-context capability

**What is LongBench?**
- Bilingual (English/Chinese) benchmark with 20+ tasks
- Multi-document QA, long code completion, summarization
- Tests up to 200K tokens
- Focus on deep understanding and cross-document reasoning

**Setup:**
```bash
# Clone repository
git clone https://github.com/THUDM/LongBench
cd LongBench

# Install dependencies
pip install -r requirements.txt

# Run evaluation (auto-downloads datasets)
python eval.py --model hybrid_v4_8M --length 100000
```

**Test protocol:**
1. **Start at 50K tokens:** Baseline evaluation on all tasks
2. **Scale to 100K tokens:** Test sustained performance
3. **Select key tasks:** Focus on multidoc_qa, code_completion, summarization
4. **Compare against baselines:** GPT-4 (~0.75 F1), Claude (~0.72 F1)

**Metrics:**
- **F1 score** for QA tasks
- **ROUGE** for summarization
- **Code accuracy** for completion
- Cross-document reasoning capability

**Expected Results:**
- **Target:** F1 >0.6 at 100K tokens
- **Comparison:** Document vs SOTA (GPT-4, Claude, Gemini)
- **Component analysis:** RWKV vs Mamba contribution to long-context

**Acceptance Criteria:**
- [ ] Evaluation runs at 50K and 100K tokens
- [ ] F1/ROUGE scores collected for all tasks
- [ ] Degradation curve plotted (accuracy vs length)
- [ ] Comparison to baseline models documented
- [ ] Task-specific strengths/weaknesses identified
- [ ] Results documented in V4_BUILD_LOG.md

**Red Flags:**
- F1 <0.4 at 50K tokens (model struggles with moderate length)
- Sharp performance cliff beyond certain length
- Specific task failures (e.g., code but not QA)

---

### Task 27: InfiniteBench Evaluation (Ultra-Long Context)

**Status:** ⬜ PENDING  
**Time:** ~4-8 hours  
**Scope:** Test "infinite-like" memory at 100K-200K tokens

**Prerequisites:**
- Task 26 (LongBench analysis) must show F1 >0.6 at 100K tokens
- Adequate hardware (80GB+ VRAM or gradient checkpointing setup)

**What is InfiniteBench?**
- Ultra-long context benchmark (100K-200K+ tokens)
- Complex tasks: book QA, infinite math sequences, full novel summarization
- Probes sustained attention and hierarchical memory
- Tests if model maintains state at extreme lengths

**Setup:**
```bash
# Follow arXiv instructions (https://arxiv.org/abs/2402.13718)
# Generate test data
python generate_infinitebench_data.py --task book_qa --length 150000

# Run evaluation
python eval_infinitebench.py --model hybrid_v4_8M --task book_qa
```

**Test protocol:**
1. **Book QA:** Answer questions about full novels (150K+ tokens)
2. **Math sequences:** Maintain state through infinite series
3. **Code repository:** Navigate entire codebases
4. **Multi-session dialogue:** Track hundreds of conversation turns

**Resource requirements:**
- **VRAM:** 80GB+ recommended (or use gradient checkpointing)
- **Time:** Hours per evaluation task
- **Consider:** Model quantization or memory-efficient attention

**Expected challenges:**
- Extreme VRAM requirements at 200K+ tokens
- Long evaluation time (hours per task)
- May require distributed inference or CPU offloading

**Acceptance Criteria:**
- [ ] Evaluation completes at 100K-150K tokens
- [ ] Accuracy measured for book QA and math sequences
- [ ] Memory usage profiled and documented
- [ ] Graceful degradation observed (not catastrophic failure)
- [ ] RWKV/Mamba component contributions analyzed
- [ ] Results documented in V4_BUILD_LOG.md

**Red Flags:**
- OOM errors below 100K tokens (inadequate memory management)
- Catastrophic accuracy drop (not graceful degradation)
- Specific task complete failures

**See Also:** [V4.5_OPTIMIZATION.md](V4.5_OPTIMIZATION.md) - Advanced long-context benchmarks section

---

### Task 28: Long-Context Optimization Pass

**Status:** ⬜ PENDING  
**Time:** ~1-2 weeks  
**Scope:** Systematic optimization based on LongBench/InfiniteBench findings

**What to optimize:**
1. **Memory efficiency:** Gradient checkpointing, quantization, efficient attention
2. **Component balance:** Tune RWKV/Mamba ratio for long-context tasks
3. **State management:** Optimize recurrent state handling at extreme lengths
4. **Training data:** Curate long-context training examples

**Approach:**
- Controlled experiments (one variable at a time)
- Document each optimization's impact
- Maintain validation loss within ±5% of baseline
- Focus on 80K-100K token range (practical use case)

**Deliverables:**
- [ ] Memory-optimized model configuration
- [ ] Long-context training data pipeline
- [ ] Performance vs memory tradeoff analysis
- [ ] Production deployment guidelines
- [ ] Updated V4_BUILD_LOG.md with optimization results

**See Also:** [V4.5_OPTIMIZATION.md](V4.5_OPTIMIZATION.md) - Full optimization methodology

---

## Models Reference

**All models use ParallelHybridBlock (RWKV6 + Mamba2 in every block)**

| Model | Layers | Hidden | Approx Params |
|-------|--------|--------|---------------|
| Hybrid-5M | 8 | 128 | ~5.2M |
| Hybrid-6M | 10 | 160 | ~6.5M |
| Hybrid-8M | 12 | 192 | ~8.0M |

**Phase 2 experiments:** Vary the rwkv_gain/mamba_gain initialization or ratio.  
**Phase 3 experiments:** Test different fusion mechanisms (concat, gated, etc).

**Architecture:** Parallel Hybrid Blocks (see V4_DESIGN.md)

---

## Key Diagnostic: Component Gradient Logging

```python
def log_component_gradients(model):
    """
    For Sequential Sandwich architecture.
    RWKV6 params are in layers 0 and 22.
    Mamba2 params are in layers 1-21.
    """
    rwkv_grad_norms = []
    mamba_grad_norms = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Check layer type by parameter name
            if 'rwkv' in name.lower():
                rwkv_grad_norms.append(param.grad.norm().item())
            elif 'mamba' in name.lower():
                mamba_grad_norms.append(param.grad.norm().item())
    
    rwkv_avg = sum(rwkv_grad_norms) / len(rwkv_grad_norms) if rwkv_grad_norms else 0
    mamba_avg = sum(mamba_grad_norms) / len(mamba_grad_norms) if mamba_grad_norms else 0
    ratio = rwkv_avg / (mamba_avg + 1e-9)
    
    # Gate G4 check
    if ratio < 0.1 or ratio > 10:
        print(f"⚠️ GATE G4 FAIL: Gradient ratio {ratio:.2f} - one component may be dead!")
    elif ratio < 0.3 or ratio > 3:
        print(f"⚠️ GATE G4 WARN: Gradient ratio {ratio:.2f}")
    
    return {'rwkv': rwkv_avg, 'mamba': mamba_avg, 'ratio': ratio}
```

---

*Start with Task 1. Do not skip ahead.*
