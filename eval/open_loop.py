"""Open-loop prediction eval: condition on k context frames + the TRUE action sequence,
roll the PRIOR forward, decode, compare predicted vs ground-truth future.

Mostly yours -- the metric is where world-model understanding shows.

Concept:  Prediction quality vs. horizon. Error grows with horizon; action-conditioning
          should help a lot (if it doesn't, you trained a video autoplayer, not dynamics).
Question: Why does action-conditioning matter for THIS metric specifically?

----------------------------------------------------------------------------------------
WHY this is the test that proves we learned DYNAMICS:
  `imagine` rolls the PRIOR p(z|h) forward with NO observations -- exactly the regime the
  policy will train in. If the model truly captured dynamics, feeding the TRUE actions should
  predict the future far better than feeding NO actions, because the only thing that makes the
  future differ from "nothing happens" is the action (in DummyEnv, pos integrates throttle).
  If true-action and no-action errors are the same, the model ignored the action -> it learned
  to autoplay the average future, not to simulate.

ACTION TIMING (continues the §3 convention, see world_model.assemble_loss):
  After observing context frames 0..c-1 the state is (h_{c-1}, z_{c-1}); the action a_{c-1}
  has NOT been consumed yet. So the first imagine step consumes a_{c-1} and produces feat_c,
  which predicts obs_c. Hence imagine actions = actions[:, c-1 : c-1+H] and the prediction
  targets are obs[:, c : c+H].
"""
import torch


def _decode_obs(model, feat):
    B, H, F = feat.shape
    return model.decoder(feat.reshape(B * H, F))["obs"].reshape(B, H, -1)


def open_loop_eval(model, batch, context=5, horizon=20):
    """Returns per-horizon-step prediction error (sum over obs dims, mean over batch) for:
      - "model":      rolled with the TRUE actions (action-conditioned).
      - "no_action":  rolled with ZERO actions through the same model (the ablation baseline).
      - "repeat_last": trivial persistence -- predict the last context obs forever.
    Each is a length-`horizon` array. The shape vs horizon is the story."""
    obs, actions = batch["obs"], batch["action"]
    B, T = obs.shape[:2]
    assert T >= context + horizon, f"need seq_len >= context+horizon ({context+horizon}), got {T}"
    rssm = model.rssm

    with torch.no_grad():
        # 1) Encode the context frames and run the posterior to get the start state.
        ctx = obs[:, :context]
        embeds = model.encoder(ctx.reshape(B * context, *obs.shape[2:])).reshape(B, context, -1)
        state = rssm.observe(embeds, actions[:, :context],
                             rssm.initial_state(B, obs.device))["state"]

        # 2) Roll the PRIOR forward `horizon` steps -- once with true actions, once with zeros.
        #    Use the prior MEAN (sample=False), NOT samples: this is a PREDICTION metric, and
        #    sampling injects noise on the (unpredictable) obs dims that swamps the signal.
        #    [Learned the hard way -- see experiments/003: with samples, true-action and
        #     no-action errors were indistinguishable; with the mean they separate cleanly.]
        img_actions = actions[:, context - 1: context - 1 + horizon]
        feat_true = rssm.imagine(img_actions, state, sample=False)["feat"]
        feat_zero = rssm.imagine(torch.zeros_like(img_actions), state, sample=False)["feat"]

        # 3) Decode predicted obs and compare to ground truth, per horizon step.
        target = obs[:, context: context + horizon]                 # (B, H, state_dim)
        last_ctx = obs[:, context - 1: context].expand(-1, horizon, -1)  # persistence prediction

        def per_h(pred):
            # sum over obs dims (so the predictable `pos` dim isn't washed out by 34 noise
            # dims), mean over batch -> one number per horizon step.
            return ((pred - target) ** 2).sum(-1).mean(0).cpu().numpy()

        return {
            "horizon": list(range(1, horizon + 1)),
            "model": per_h(_decode_obs(model, feat_true)),
            "no_action": per_h(_decode_obs(model, feat_zero)),
            "repeat_last": per_h(last_ctx),
        }
