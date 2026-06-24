# Roadmap — Improving the Learned Driving Policy

Forward plan for raising the policy's driving accuracy/ability, after the obs→action ablation
reframed the project. Living document — update status as items ship.

## Where we are (the reframing finding)
The **direct obs→action policy is our base** — not the world-model latent.

| policy | route | off-road | return | bc_loss | verdict |
|---|---|---|---|---|---|
| **DIRECT-BC** (plain MLP, no WM) | **22%** | 90% | +64 | 0.05 | drives, but drifts off |
| LATENT-BC (WM + actor) | 2% | 0% | +2 | 2.06 | idles (couldn't fit the expert) |
| RANDOM | 2% | 0% | +1.5 | — | baseline |

(`scripts/ablate_direct_bc`, n=10, held-out maps.) The under-trained world-model latent loses the
lane/heading signal, so every latent-based method (BC, DAgger, RL) idled. Cloning directly from the
259-dim state works. **The number to attack is off-road 90% = behavior-cloning distribution shift:**
the policy commits to driving but, having only seen clean centerline demos, can't recover once it drifts.

## Tier 1 — highest leverage (build next)

### A. Recovery data via perturbation  ← ✅ **DONE — it works**
**Result (`scripts/recovery_bc`, n=10 held-out):** clean+recovery vs clean-only →
route **24% → 39%**, crash **20% → 0%**, off-road **50% → 30%**, return +93 → +138, steps 137 → 174.
A real win (gap > the ±11% std). Best learned policy so far; saved `runs/direct_bc/policy_recovery.pt`.

The off-road fix. The policy never learned recovery because it only saw the expert driving cleanly.
Inject perturbations during expert collection (DART; Codevilla §A.3; ChauffeurNet) so the car visits
**off-center states**, and record **IDM's correct action there** as the label — recovery is now *in*
the training set.
- **Mechanism (chosen):** drive the car ourselves; each step query `IDMPolicy.act()` as the clean
  **label**, and *execute* `label + steering perturbation` so the car drifts. (Reuses the proven
  manual-IDM-query from DAgger; avoids touching vehicle pose internals.)
- **Perturbation (Codevilla Eq 3):** triangular steering impulse
  `s(t) = σ·γ·max(0, 1 − |2(t−t₀)/τ − 1|)`, started with prob `p`/step, `σ∈{−1,+1}`, `γ≈0.15`,
  `τ∈[0.5,2]s`. Pure + unit-tested.
- **Train** direct-BC on clean demos **+** recovery demos; eval on held-out. Success = off-road ↓,
  route ↑ vs clean-only.
- Files: `training/recovery.py` (`triangular_impulse`, `collect_idm_perturbed`),
  `scripts/recovery_bc.py` (clean vs clean+recovery comparison).

### B. Eval gauge — finish it (so we can measure A)
- ✅ per-episode variability (`route X%±Y%`, `return ±Z`) — shipped.
- ☐ **Deterministic fixed eval seeds** (same held-out maps every run → comparable).
- ☐ **Targeted recovery metric**: perturb the first N steps off-center, measure recovery rate —
  isolates exactly the failure A targets, better than aggregate off-road %.

## Tier 2 — after A shows lift
### C. More + denser demos
Direct-BC fit easily (bc_loss 0.05) → capacity to spare; coverage is the limit. Raise `collect_steps`.

### D. Auxiliary off-road / progress losses (ChauffeurNet's other half)
Add explicit terms beyond cloning: penalize actions that lead off-road, reward progress. Needs the
perturbation data (A) for signal.

## Tier 3 — diminishing returns / more complex
### E. Advanced objectives
- Closed-loop / multi-step rollout loss (penalize *compounding* error, not per-step MSE).
- Action-distribution head (mixture/quantile/diffusion) for a multimodal expert (pass left vs right).
- Input normalization of the 259 dims (lidar/nav/ego live on different scales).

## Honest ceiling
IDM (route 99%) is the teacher, so a clone caps *near* IDM minus distribution-shift losses. Realistic
target: route climbing toward ~50–70%+ and `success_rate` finally > 0% — not 99%. CPU-bound, so keep
budgets modest.

## Deprioritized (the ablation made these not worth it for *policy* accuracy)
- World-model objective swaps (R2-Dreamer / decoder-free TD-MPC2) — we bypass the latent now.
- More DAgger tuning (β-mixing, etc.) — offline perturbation (A) is the simpler recovery source.
- These matter only if we later pursue imagination-based RL, which is not the current path.
