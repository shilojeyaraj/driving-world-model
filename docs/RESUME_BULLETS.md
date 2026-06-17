# Resume bullets — Driving World Model

LaTeX-formatted to match the existing resume style (`\projectHeading` / `\toolsline` /
`resume_list`). Every number is real and traceable to the repo (see notes at the bottom).

---

## Full block (drop-in)

```latex
\projectHeading{Driving World Model — From-Scratch Dreamer for Model-Based RL}{}{https://github.com/shilojeyaraj/driving-world-model}
\toolsline{ PyTorch, Python, NumPy, Variational Inference (ELBO), Model-Based RL, RSSM, Actor--Critic, Mamba/SSM, CNNs, MetaDrive, pytest}\\[1pt]
\begin{resume_list}
  \item Built a \textbf{Dreamer-style world model for autonomous driving from scratch} in PyTorch (\textbf{${\sim}$3{,}700 LOC}, \textbf{55 tests}), training a control policy \textbf{entirely inside the model's learned ``imagination'' with zero real-environment steps}, which drove the task near-optimally at a return of \textbf{${\approx}95$ vs. ${\approx}{-}51$} for a random baseline.
  \item Implemented the full variational training stack --- a recurrent state-space model (\textbf{RSSM}) trained on the \textbf{ELBO} with the reparameterization trick and \textbf{free-bits regularization} --- to learn latent driving dynamics, \textbf{eliminating posterior collapse} and reducing pixel reconstruction loss from \textbf{245 $\to$ 0.17} ($\sim$\textbf{1{,}400$\times$}) so the model could ``dream'' coherent future video frames.
  \item Engineered a \textbf{controlled ablation} of the dynamics core behind a single one-method interface --- a \textbf{GRU-based RSSM vs. a Mamba-style selective state-space model} --- swappable via one config flag and re-scored on an identical open-/closed-loop eval harness across \textbf{17 logged experiments}.
  \item Integrated the \textbf{MetaDrive} driving simulator behind a unit-tested observation contract, training the world model on real \textbf{259-dimensional lidar + ego-state} inputs, and documented exactly where the simplified stack succeeds vs. hits known model-based RL failure modes (model exploitation, actor corner-collapse).
\end{resume_list}
```

---

## Optional 5th bullet (debugging depth — strong interview hook)

```latex
  \item Diagnosed and fixed two subtle correctness bugs --- an \textbf{action-timing off-by-one} and a \textbf{stochastic-vs-mean measurement error} --- using held-out generalization tests and instrumentation rather than guesswork, establishing deterministic prediction metrics that made action-conditioning measurable.
```

---

## Trimmed 2-bullet version (tight resume)

```latex
\projectHeading{Driving World Model — From-Scratch Dreamer for Model-Based RL}{}{https://github.com/shilojeyaraj/driving-world-model}
\toolsline{ PyTorch, Python, Model-Based RL, RSSM, Variational Inference (ELBO), Actor--Critic, Mamba/SSM, MetaDrive, pytest}\\[1pt]
\begin{resume_list}
  \item Built a \textbf{Dreamer-style world model for driving from scratch} in PyTorch (\textbf{${\sim}$3{,}700 LOC}, \textbf{55 tests}), training a policy \textbf{purely in the model's imagination with zero env steps} to a return of \textbf{${\approx}95$ vs. ${\approx}{-}51$} random; built on an \textbf{RSSM} trained via the \textbf{ELBO} (reparameterization + free bits), cutting reconstruction loss \textbf{245 $\to$ 0.17}.
  \item Ran a \textbf{controlled GRU-vs-Mamba dynamics ablation} behind one swappable interface and integrated the \textbf{MetaDrive} sim (real \textbf{259-dim} lidar+ego obs) across \textbf{17 logged experiments}, documenting where the from-scratch stack succeeds vs. known model-based RL failure modes.
\end{resume_list}
```

---

## Where each number comes from (so you can defend it in an interview)

| Stat | Source |
|------|--------|
| ~3,700 LOC | `wc -l` over all `*.py` |
| 55 tests / 19 test files | `tests/` |
| 17 experiments | `experiments/001–017` logs |
| return ≈95 vs ≈−51 | `README.md`, `docs/SYSTEM_OVERVIEW.md` §7 (toy closed-loop) |
| recon 245 → 0.17 | `docs/SYSTEM_OVERVIEW.md` §7 (visual/real-sim learning) |
| 259-dim obs | MetaDrive `state_dim=259`, `docs/RUNNING.md` §7 |
| GRU vs Mamba ablation | `models/recurrence.py`, `scripts/ablate_dynamics.py`, exp 006 |
| posterior collapse / free bits | `models/rssm.py`, `ARCHITECTURE.md` |

**Notes**
- This is a solo, from-scratch research/engineering project — there is intentionally **no user/scale
  claim** (that would be dishonest for this kind of project).
- The GitHub URL is a placeholder (`shilojeyaraj/driving-world-model`) — fix it to the real repo, or
  delete the third `\projectHeading` argument if the repo is private.
- `${\sim}$` renders as “~”, `${\approx}$` as “≈”, `$\to$` as “→”, `$\times$` as “×”. If your resume
  preamble already defines these, swap to whatever macros you use.
```
