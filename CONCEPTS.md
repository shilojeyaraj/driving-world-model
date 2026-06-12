# Concept map

Each load-bearing file teaches a specific ML concept. Implement it from scratch, then make
sure you can answer its question without notes. If you can't, you haven't learned it yet.

| File | Build | Concept it teaches | Question to answer |
|------|-------|--------------------|--------------------|
| `models/encoder.py` | from scratch | Representation learning; conv inductive bias vs. attention (CNN vs ViT) | Why might a ViT need more data than a CNN for the same reconstruction quality? |
| `models/rssm.py` | from scratch | Latent-variable sequence models; the ELBO; prior vs posterior; KL; reparameterization | Where does `KL(posterior‖prior)` come from, and what does the prior do at imagination time? |
| `models/decoder.py` | from scratch | Likelihood/reconstruction heads; (optional) diffusion (DiT) | Why predict reward and continue, not just pixels? |
| `models/world_model.py` | from scratch (loss) | Assembling the variational objective; KL balancing; free bits | What is posterior collapse, and how do you detect it from the loss curves? |
| `models/actor_critic.py` | from scratch | Actor-critic RL; value functions; λ-returns; model-based vs model-free | Why can you train the policy on imagined rollouts with zero env steps? |
| `data/replay_buffer.py` | library ok | Sequence sampling for sequence models | Why sample contiguous sequences, not i.i.d. transitions? |
| `envs/*.py` | library ok | Env interface; observation design | What's in a "state" obs vs an "image" obs, and why does that change compute cost? |
| `eval/open_loop.py` | mostly yours | Prediction-quality metrics; horizon decay | Why does action-conditioning matter for this metric? |
| `eval/closed_loop.py` | mostly yours | Task metrics; the two eval axes | Great open-loop but bad closed-loop -- what went wrong? |

## Learning protocol (per component)

1. **Predict first.** Before running, write what you expect and *why* (in the experiment log).
2. **Implement from scratch.** No copy-paste of the core math.
3. **Derive once.** For the RSSM, work the ELBO out by hand. If you can derive it, you own it.
4. **Ablate.** Remove the KL term, drop action-conditioning -- watch what breaks. That's the lesson.
5. **Write the failure down.** "Posterior collapsed; caught it via flat KL + blurry recon" is
   the sentence that proves understanding in an interview.
