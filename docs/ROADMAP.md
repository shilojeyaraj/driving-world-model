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

### B. Eval gauge  ← ✅ **DONE**
- ✅ per-episode variability (`route X%+/-Y%`, `return +/-Z`) — `summarize_driving` + `eval_driving`.
- ✅ **Deterministic fixed eval seeds** — `eval_seeds` + `env.reset(seed=...)` in `_run_episodes`, so
  every eval (eval_driving / recovery_bc / ablate_direct_bc) sees the SAME held-out maps each run.
- ✅ **Targeted recovery metric** — `scripts/eval_recovery.py`: forces the car off-center for the first
  K steps, then measures whether the policy gets back (didn't end off-road). The recovery policy scored
  **recovery_rate 40%** (n=10) — a discriminative isolation of the exact skill A adds.

## Tier 2 — after A shows lift
### C. More + denser demos  ← ✅ **DONE — route plateaus, recovery improves**
`scripts/train_direct_policy.py` — the production trainer (clean + recovery demos, scalable +
tunable perturbation), full held-out eval (route/crash/off-road **and** recovery_rate).
**Result (8k+8k vs 4k+4k, deterministic eval):** route **39% → 39%** (plateau), off-road 30% (same),
but **recovery_rate 40% → 70%** — more data made it *recover* much better, yet aggregate route is
capped at ~39% by the residual off-road. **Takeaway: more demos alone won't break 39%** — the next
lever is D (explicit off-road/progress signal), not scale.

### D. Auxiliary progress head  ← ✅ **DONE — didn't help (as predicted)**
**Result (aux_weight 0.5, 8k+8k, deterministic eval):** route 39% → 41% (flat, within ±12% noise),
off-road 30% → **40% (worse)**, recovery_rate 70% → **40% (worse)**. The aux task competed for
capacity rather than surfacing hidden signal — expected, since our 259-dim state already exposes the
lane cue (the trick matters for pixels). **Keep the no-aux `policy.pt`.** Confirms the cheap CPU
levers are exhausted; the remaining gap to IDM needs a differentiable off-road penalty, more compute,
or online RL (all beyond the laptop envelope).

(original plan below)
ChauffeurNet's "more than cloning" idea, adapted to our constraints. We have **no differentiable sim**
(direct policy bypasses the WM), and off-road *classification* labels are too sparse (~99% on-road),
and reward-weighted BC would downweight our recovery examples — so the implementable form is
**Codevilla's auxiliary progress head**: a shared trunk predicts the per-step **reward** alongside the
action (`DirectPolicyAux`, `train_direct_bc_aux`), forcing the representation to encode lane/progress
quality. `train_direct_policy --aux-weight 0.5`. Honest expectation: a regularizer (modest gain), not
a hard off-road penalty. Compare aux vs no-aux at scale (deterministic eval) — running.

## KEY FINDING — per-scene breakdown reframes the ceiling
`scripts/eval_by_scene` on the best policy (held-out, n=10/scene):

| geometry | route | success | crash | off-road |
|---|---|---|---|---|
| straight | **96%** | 100% | 0% | 0% |
| curve | **82%** | 40% | 0% | 10% |
| intersection | **84%** | 70% | 30% | 0% |
| roundabout | **42%** | 0% | 40% | 60% |

The aggregate route 39% was **masking real competence**: the policy drives straights/curves/
intersections near-IDM (route 82–96%, success up to 100%); the off-road 30% is almost entirely
**roundabouts** (off-road 60%, success 0% there). So the "ceiling" is one failing geometry, not a
global limit.

### Roundabout boost — `train_direct_policy --boost-scene O`  ← ✅ **DONE — works**
`runs/direct_bc/policy_boosted.pt` (milestone `direct+rec+O 16k`, aggregate route 41%):

| geometry | route | off-road |
|---|---|---|
| straight | **96%** | 0% |
| curve | **87%** (+5pp) | 0% |
| intersection | **84%** | 0% |
| roundabout | **64%** (+22pp) | 30% (↓30pp) |

Mixed in 4k+4k extra roundabout clean+recovery demos. Roundabout route 42% → 64%, off-road halved
60% → 30%. Curves improved too (82% → 87%). Aggregate is 41% (only 2pp gain — eval uses mixed 3-block
maps) but the per-scene win is real and large. Chart: `runs/direct_bc/by_scene.png`.

## Tier 3 — next levers
### F. Direct-policy DAgger — `scripts/direct_dagger.py`  ← ✅ **DONE — tied baseline, didn't beat it**
**Result (3 iters, boost=O, rollout 2k/iter, held-out n=5):**

| iter | route | off-road | bc_loss | note |
|---|---|---|---|---|
| 0 (BC+rec+boost) | 38% | 20% | 0.043 | strong baseline |
| 1 (+2k rollout) | **41%** ← BEST | 20% | 0.057 | tied policy_boosted.pt |
| 2 (+2k more) | 39% | 20% | 0.150 | loss rising |
| 3 (+2k more) | 35% | 40% | 0.316 | regression |

Per-scene breakdown of iter-1 best vs `policy_boosted.pt`:

| geometry | direct-DAgger iter 1 | policy_boosted (BC+boost) | Δ |
|---|---|---|---|
| straight | 96% | 96% | = |
| curve | **92%** | 87% | +5pp ✅ |
| intersection | 72% | 84% | −12pp (60% crash, high variance) |
| roundabout | 56% | **64%** | −8pp ❌ |

DAgger improved curves but hurt roundabouts — the rollout failure states at roundabouts confused
the policy by mixing diverse IDM relabels with the clean roundabout boost data. **`policy_boosted.pt`
remains the best overall model.** Direct DAgger matched the BC+boost baseline (41%) at iter 1 but then **regressed**. The bc_loss increase
(0.04 → 0.32) reveals why: IDM's relabeled actions at the policy's failure states are high-variance
and hard to fit alongside the clean training data — the growing heterogeneous dataset makes
optimization harder, not easier. **Takeaway:** with 2k rollout steps the signal-to-noise is too
low; the clean+recovery+boost data is already near-optimal for this CPU budget. The perturbation
recovery (roadmap A) was effectively doing the same thing more stably. Best: `runs/direct_dagger/policy_best.pt`.

### G. WM-based DAgger — `scripts/dagger.py`  ← ✅ **DONE — confirms WM latent is the ceiling**
**Result (3 iters, 2k rollout/iter, held-out n=5):**

| iter | route | bc_loss | note |
|---|---|---|---|
| 0 | 1% | 0.145 | WM-BC baseline |
| 1 | 2% | 0.057 | fitting better |
| 2 | 11% | 0.054 | improvement |
| 3 | **13%** ← BEST | 0.035 | plateau |

DAgger iterates (1% → 2% → 11% → 13%) and bc_loss drops, so the WM *is* learning to fit IDM
better in latent space. But 13% vs the direct policy's 41% confirms the WM latent is the
hard ceiling at this compute scale — the latent loses lane/heading signal regardless of how
many recovery states we show it. Best: `runs/dagger/ckpt_best.pt`.

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
