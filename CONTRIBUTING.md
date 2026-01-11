# Contributing to GroundThink

**Welcome!** This guide explains how to contribute to GroundThink's hybrid architecture research.

**Current Status:** V0.5 (Twin Debate) planning phase. V4 graduated with GPT-2 parity. See [ABOUT.md](ABOUT.md) for project overview.

---

## Documentation & Versioning SOP

### The Problem We Solved

Session-dated files, redundant docs, orphaned code, and stale meta-documents accumulated because there were no rules. This SOP prevents that.

### Single Source of Truth

| What | Where | NOT Here |
|------|-------|----------|
| Current version | `VERSION` | ~~VERSIONS.md, VERSION_REGISTRY.md~~ |
| Version history | `CHANGELOG.md` | ~~V4_BUILD_LOG.md~~ |
| V4 task backlog | `V4_STRATEGY.md` (archived summary) | ~~session summaries~~ |
| V0.5 roadmap | `V0.5_ROADMAP.md` | ~~scattered docs~~ |
| Current status | `V4_HANDOFF.md` | ~~V4_STRATEGY.md header~~ |
| Test definitions | `V4_TESTING.md` + `CANARY_TESTS.md` | ~~scattered in strategy~~ |

### Document Creation Rules

| Situation | Action | Wrong Action |
|-----------|--------|--------------|
| New task/phase | Add to `V4_STRATEGY.md` | ❌ Create PHASE_X_SUMMARY.md |
| Session notes | Add to commit message | ❌ Create SESSION_2026_01_10.md |
| Audit findings | Update relevant doc | ❌ Create AUDIT_COMPLETE.md |
| New metric | Add to `V4_TESTING.md` | ❌ Create METRICS.md |
| Completed phase | Update `CHANGELOG.md` + `VERSION` | ❌ Create PHASE_COMPLETE.md |

### Version Bump Protocol

```
When to bump VERSION:
- Phase completion → Minor bump (4.0 → 5.0)
- Architecture change → Version update (V4 → V0.5)
- Significant milestone → Patch bump (5.0 → 5.1)
- Breaking change → Major bump

VERSION format:
  X.Y-Status (Version Z Architecture)
  
Example:
  5.0-Alpha (V0.5 Twin Debate Architecture)
```

**Always update CHANGELOG.md when bumping VERSION.**

### Archive Policy

| Condition | Action |
|-----------|--------|
| Content moved elsewhere | Archive with commit note citing destination |
| Session-dated file | Archive immediately after session ends |
| No references for 30+ days | Archive with "orphaned" note |
| Superseded by consolidation | Archive, update refs in remaining docs |

**Never archive:** README.md, CHANGELOG.md, LICENSE, CONTRIBUTING.md, VERSION

**Archive path:** `archive/[original-filename]`

### Locked Terminology

| Term | Meaning | NOT This |
|------|---------|----------|
| V5 | Gate (blocker for 8M scaling) | Not a version bump |
| Phase 4.0 | 3.5M validation complete | — |
| GF-MH | Gated Fusion Mamba-Heavy | Not "hybrid" or "fusion model" |
| Tasks 62-66 | V5 gate prerequisites | Not "Phase 5 tasks" |
| cuda_backends.py | CUDA kernel hub | Not "FLA" (FLA removed) |
| CER | Compute-Efficiency Ratio | — |
| UCW | Useful Context Window | — |
| SPS | State Persistence Score | — |

**Rule:** Use locked terms consistently. If you need a new term, add it here first.

### Before Creating a New Document

1. **Check existing docs first** — 80% of content belongs in CHANGELOG, V4_STRATEGY, or existing doc
2. **If new doc needed:**
   - Add entry to this table (Single Source of Truth)
   - Include "Owner" (who updates) and "Archive Trigger" in header
3. **Session-dated files** (e.g., NOTES_2026_01_10.md) go directly to `archive/`

### Commit Hygiene

| Trigger | Action |
|---------|--------|
| "SOP commit" | Run ref check → Update CHANGELOG + VERSION → commit → push |
| "SOP review" | Run ref check → Update CHANGELOG + VERSION → show diff → wait |
| Session end | Always push before closing |
| Logical milestone | Commit immediately (don't batch unrelated changes) |

**Before pushing, run:**
```bash
./tools/check_refs.sh  # Validates cross-references
```

### Handoff Document Rules

`V4_HANDOFF.md` must stay under 100 lines. It contains:
1. Current version/phase (1 line)
2. Last session summary (5 lines max)
3. Next actions (5 lines max)
4. Quick start commands (10 lines)
5. Links to detailed docs

**If it grows past 100 lines:** Content belongs elsewhere. Move to V4_STRATEGY.md or create appropriate doc.

### Code File Rules

| Location | Purpose | Examples |
|----------|---------|----------|
| Root | Entry points, utilities | train_v4.py, cuda_backends.py |
| `models/` | Model variants | hybrid_v4_GF.py |
| `tests/` | Test harnesses | test_tiny_graduation.py |
| `tools/` | Utility scripts | benchmark_suite.py |
| `ops/` | CUDA operations | (future consolidation) |
| `archive/` | Deprecated code | train.py, layers.py |

**Rule:** If a file imports from archived files, it should also be archived.

---

## Document Naming Convention

| Prefix | Purpose | Example |
|--------|---------|---------|
| `V4_` | Core documentation (current architecture) | V4_DESIGN.md, V4_STRATEGY.md |
| `V4.5_` | Lessons learned for future versions | V4.5_CUDA_KERNELS.md |
| `V5_` | Next major version planning | V5_GATING.md |
| No prefix | Standard files (README, CHANGELOG, etc.) | CHANGELOG.md, VERSION |
| `archive/` | Superseded documents (keep for reference) | archive/V4_BUILD_LOG.md |

**Rule:** V4.5 documents are "notes for V5" — they capture optimization research and implementation learnings that may be useful when building the next major version.

---

## Before You Start

1. **Read [ONBOARDING.md](ONBOARDING.md)** — Understand RWKV, Mamba, and the hybrid hypothesis
2. **Understand the current state** — Read [V4_HANDOFF.md](V4_HANDOFF.md) for quick status
3. **Know what we need** — Check [V4_STRATEGY.md](V4_STRATEGY.md) for Phase 4.0+ tasks

---

## Types of Contributions

### 🔬 Research Contributions (New Variants, Benchmarks)

**Example:** Testing a new fusion strategy or training approach

**Process:**
1. Fork the repository
2. Create a new file following the pattern: `hybrid_v4_[NAME].py`
   - Copy from [hybrid_v4_GF.py](hybrid_v4_GF.py) as template
   - Implement your fusion/architecture change
   - Keep the same interface (forward pass signature)
3. Add your variant to [benchmark_variants.py](benchmark_variants.py)
4. Run benchmark: `python benchmark_variants.py`
5. Create a PR with:
   - Name: `feat(variant): [Your fusion/approach name]`
   - Description: Why it might work, comparison to GF-MH
   - Results table (val loss, throughput, training stability)
6. If it beats GF-MH, we merge it and update the documentation

**Complexity Assessment:** Use [V4_STRATEGY.md Task Assessment Matrix](V4_STRATEGY.md#task-assessment-matrix)
- Simple variant (different weights): **M** (30m-2h)
- Novel fusion strategy: **L** (2-6h)
- New architecture (e.g., attention module): **XL** (6h+, break it down)

### 📋 Documentation Contributions (Fixes, Clarifications)

**Example:** Finding a gap in ONBOARDING.md, fixing a broken link, clarifying terminology

**Process:**
1. Identify the issue (broken reference, unclear explanation, outdated info)
2. Check which file owns this concept:
   - Conceptual: [ONBOARDING.md](ONBOARDING.md)
   - Architecture: [V4_DESIGN.md](V4_DESIGN.md)
   - Tasks/Progress: [V4_STRATEGY.md](V4_STRATEGY.md)
   - Status: [V4_HANDOFF.md](V4_HANDOFF.md)
   - Implementation: Code files (hybrid_v4_*.py)
3. Make the fix
4. Commit with `docs:` prefix: `docs: Fix typo in ONBOARDING.md Part 2`
5. No PR needed for typos/clarity fixes (maintainers will review and merge quickly)

**SOP:** Follow the [V4_STRATEGY.md Librarian guidelines](V4_STRATEGY.md#librarian-agent-role) — if you improve a document, document why you changed it

### 🐛 Bug Reports

**Example:** Code crashes, incorrect results, import errors

**Process:**
1. Check [CHANGELOG.md](CHANGELOG.md) for known issues
2. Run [test_phase0_complete.py](test_phase0_complete.py) to verify environment
3. Open an issue with:
   - Exact error message and traceback
   - Steps to reproduce
   - Your environment (Python version, CUDA version, OS)
   - Which file failed (variant code, training script, etc.)

**Quick Fix:** If it's an import error, check [V4_HANDOFF.md AUDIT SUMMARY](V4_HANDOFF.md#audit-summary) for known issues

### ⚡ Performance/Optimization Contributions

**Example:** Speeding up training, reducing memory usage, better monitoring

**Process:**
1. Baseline your change: Run benchmark on main branch, record metrics
2. Apply optimization
3. Benchmark again, compare
4. Document the change in [V4.5_OPTIMIZATION.md](V4.5_OPTIMIZATION.md)
5. Create PR with before/after numbers

**Complexity:** Usually **L** or **XL** (optimization requires careful testing)

---

## The Workflow (Standard Flow)

### For Code Changes
```bash
# 1. Clone and setup
git clone https://github.com/9to5ninja/groundthink.git
cd groundthink
source .venv/bin/activate
pip install -r requirements.txt

# 2. Create your branch
git checkout -b feat/my-variant  # or docs/my-fix, or fix/bug-name

# 3. Make your changes
# ... edit files ...

# 4. Test locally
python benchmark_variants.py  # If adding a variant
python test_phase0_complete.py  # Sanity check

# 5. Commit with a message following the pattern:
git commit -m "feat(variant): GF-Weighted - learnable per-layer weights

- Implements weighted fusion with per-layer parameters
- Tested on 500 steps, batch_size=64
- Val loss: 1.672 (-0.2% vs GF-MH)

Variant comparison in: results/gf_weighted_bench.txt"

# 6. Push and create PR
git push origin feat/my-variant
# Open PR on GitHub
```

### For Documentation Changes
```bash
# 1. Follow same setup as above

# 2. Edit the doc
# vim ONBOARDING.md
# OR
# vim V4_DESIGN.md

# 3. Test links (manual: click a few)
# Or if you wrote an example, test it

# 4. Commit
git commit -m "docs(onboarding): Add section on gradient debugging

- Added 'Debugging Gradient Issues' subsection to Part 7
- References V4_BUILD_LOG.md Session 9 troubleshooting
- Clarifies how to spot imbalanced components"

# 5. Push for review
git push origin docs/my-fix
```

---

## Code Review Expectations

### What We Look For

**Research Contributions:**
- ✅ Fair comparison (same data, same training steps as GF-MH)
- ✅ Stability (no crashes, NaN, exploding gradients)
- ✅ Clear reasoning (why you think it will work)
- ✅ Results table (val loss, throughput, training time)

**Documentation Contributions:**
- ✅ Accuracy (matches current code state)
- ✅ Clarity (understandable to beginners)
- ✅ Completeness (links to related docs, no dead references)
- ✅ Consistency (matches terminology in ONBOARDING.md)

**Optimization Contributions:**
- ✅ Measured improvement (numbers, not anecdotes)
- ✅ No regression (other benchmarks don't get slower)
- ✅ Documented (what changed and why)

### What Might Cause a Rejection

- ❌ "Interesting idea but untested" (test it first)
- ❌ Breaking existing tests or APIs
- ❌ Documentation that contradicts ONBOARDING.md
- ❌ Very long PRs without clear scope (break into smaller PRs)

---

## Using the Task Assessment Matrix

**For contributors:** Before proposing a change, ask yourself:

| Question | Answer | Complexity | Next Step |
|----------|--------|------------|-----------|
| Does it touch >3 files? | Yes | Likely L/XL | Break into smaller PRs |
| Is it vague ("research then build")? | Yes | Needs clarity | Add acceptance criteria |
| Estimated time? | >2 hours | L/XL | Propose as an issue first |
| Does it need validation gates (G1-G4)? | Yes | At least L | See [V4_STRATEGY.md](V4_STRATEGY.md) |

**Reference:** [Task Assessment Matrix in V4_STRATEGY.md](V4_STRATEGY.md#task-assessment-matrix)

---

## Example: Adding a New Fusion Variant (Full Walkthrough)

### Step 1: Understand Existing Variants
```python
# Read hybrid_v4_GF.py to see:
# - How RWKV and Mamba are called
# - How fusion happens in forward()
# - How parameters are initialized
```

### Step 2: Create Your Variant
```python
# Copy hybrid_v4_GF.py to hybrid_v4_MY.py
# Change the fusion mechanism in forward():

class ParallelHybridBlock_MY(nn.Module):
    def forward(self, x):
        # ... norm, RWKV, Mamba calls ...
        
        # YOUR FUSION HERE (replace gate fusion):
        # Example: learned per-head weighting
        rwkv_out = self.rwkv6(norm_x)[0]
        mamba_out = self.mamba2(norm_x)
        
        # MY IDEA: different weight per head
        weights = self.head_weights  # [n_heads]
        # (reshape and apply your fusion)
        
        return x + fused_output
```

### Step 3: Add to Benchmark
```python
# In benchmark_variants.py, add:
variants = {
    'MY': hybrid_v4_MY.HybridModel_MY,  # your class
    'GF': hybrid_v4_GF.HybridModel_GF,  # existing
    # ...
}
```

### Step 4: Run Benchmark
```bash
python benchmark_variants.py
# Results show: MY val loss 1.668 (-0.2% vs GF-MH)
# Note: improvement is marginal, might not be worth merge
```

### Step 5: Create PR
```
Title: feat(variant): Per-Head Weighted Fusion (MY)

Description:
- Tested per-head learned weights instead of per-position
- Hypothesis: different heads might benefit from different RWKV/Mamba blend
- Results: 1.668 val loss (-0.2% vs GF-MH, within noise margin)
- Conclusion: Not a meaningful improvement; included for reference

Benchmark results attached.
```

**Outcome:** Even if not merged, you've contributed to the exploration record. Others can see it was tried.

---

## Communication & Etiquette

### Be Specific
- ❌ "Your documentation is confusing"
- ✅ "The ONBOARDING.md Part 2 doesn't explain why Mamba's selectivity matters for long-context"

### Assume Good Intent
- This is a research project; some things are experimental
- Docs may lag code; help update them
- Variants may not beat the winner; that's still useful data

### Ask Before Large Changes
- If you want to refactor code significantly, open an issue first
- If you want to reorganize docs substantially, discuss with maintainers

---

## Getting Help

**Questions about the project?**
- Check [ONBOARDING.md](ONBOARDING.md) Part 8 (External Resources)
- Read [V4_HANDOFF.md](V4_HANDOFF.md) (current status)

**Technical questions?**
- Search [V4_BUILD_LOG.md](V4_BUILD_LOG.md) for similar issues
- Check test files (test_phase0_complete.py, test_monitoring.py)

**Understanding the code?**
- Start with [hybrid_v4_GF.py](hybrid_v4_GF.py) (simplest variant)
- Read its comments, cross-reference [V4_DESIGN.md](V4_DESIGN.md)

---

## Recognition

Contributors are recognized in:
- PR merged: GitHub author attribution
- Major contribution: Listed in repository README.md contributors section
- Sustained contribution: Consider for collaborator role

---

## License & Intellectual Property

By contributing, you agree that your work:
- Can be used under the same license as GroundThink (specify yours)
- May be included in research papers or presentations
- Is original work or properly attributed

Check the LICENSE file for full terms.

---

**Last Updated:** 2026-01-10  
**Maintained By:** GroundThink Team  
**Questions?** Open an issue on GitHub
