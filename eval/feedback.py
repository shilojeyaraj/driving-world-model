"""Driving-feedback engine (GF4): critique a driver (you, via gestures) using the world model.

Concept:  Three complementary signals, each from a part of the model-based stack:
            A. OUTCOME FORECAST  -- imagine your action forward, read the continue/reward heads:
               "where does this lead?" (uses the prior MEAN, like open-loop prediction).
            B. STYLE DEVIATION   -- compare your action to a reference policy (IDM expert):
               "how does this differ from how an expert drives here?".
            C. STATE VALUE       -- the critic's value of the state: "how good is this situation?".
Question: Why is the CONTINUE head the most trustworthy of these on a hard sim, and the reward/
          value the least? (Hint: experiments/010 -- which signal actually learned cleanly?)

The metric functions are small and pure-ish so they're unit-testable on tiny untrained models;
DrivingFeedback carries the RSSM posterior across steps exactly like eval/closed_loop.py.

Mostly yours -- the *fusion* (report_from_traces) is where "feedback on driving habits" lives.
"""
import numpy as np
import torch

# Event thresholds (deviation from the reference, in action units of [-1,1]).
_STEER_DEV = 0.5
_THROTTLE_DEV = 0.5


def forecast_safety(world_model, state, action, horizon):
    """A: imagine `action` held for `horizon` steps from the current latent `state` and read the
    continue/reward heads. Returns predicted survival (prob of staying 'alive' over the horizon),
    predicted return, and a risk flag. Uses the prior MEAN (sample=False) -- a prediction, not a
    sample (experiments/003)."""
    cfg = world_model.cfg
    device = torch.device(cfg.device)
    A = cfg.action_dim
    a = torch.as_tensor(np.asarray(action, np.float32), device=device).reshape(1, 1, A).expand(1, horizon, A)
    with torch.no_grad():
        feats = world_model.rssm.imagine(a, state, sample=False)["feat"]            # (1, H, F)
        dec = world_model.decoder(feats.reshape(horizon, feats.shape[-1]), decode_obs=False)
        cont = torch.sigmoid(dec["cont_logit"])                                      # (H,)
        survival = float(torch.prod(cont))                                           # P(alive through H)
        pred_return = float(dec["reward"].sum())
    return {"survival": survival, "pred_return": pred_return, "risk": survival < cfg.risk_threshold}


def style_deviation(reference_actor, feat, action):
    """B: how your action differs from the reference policy's at this state. Reports the per-dim
    deviation AND `surprise` = negative log-prob of your action under the reference's distribution
    (a principled 'how unusual is this' that accounts for the policy's uncertainty)."""
    a = np.asarray(action, np.float32)
    with torch.no_grad():
        ref, _ = reference_actor(feat, deterministic=True)                           # (1, A)
        surprise = -float(reference_actor.log_prob(feat, torch.as_tensor(a, device=feat.device).reshape(1, -1)))
    ref = ref.squeeze(0).cpu().numpy()
    return {"d_steer": float(a[0] - ref[0]), "d_throttle": float(a[1] - ref[1]),
            "ref_steer": float(ref[0]), "ref_throttle": float(ref[1]), "surprise": surprise}


def state_value(critic, feat):
    """C: the critic's value of the current state (how good is this situation)."""
    with torch.no_grad():
        return float(critic(feat).reshape(-1)[0])


def should_forecast(step_idx, every):
    """Whether to recompute the EXPENSIVE safety forecast (the 15-step imagine) this step. The live
    HUD reuses the last forecast in between, which lets a weak laptop run the cheap per-step signals
    (state-carry, style, value) at full fps while the costly imagine fires only every `every` frames.
    `every<=1` -> every step (no throttling)."""
    return every <= 1 or step_idx % every == 0


class DrivingFeedback:
    """Run the 3 signals online while you drive. Carries the RSSM posterior across steps exactly
    like eval/closed_loop.py: feed the PREVIOUS action + the CURRENT obs to obs_step, act on
    feat=[h;z]. `step(obs, action)` returns the per-step signals; `finalize()` aggregates them."""

    def __init__(self, world_model, reference_actor, critic, cfg):
        self.wm = world_model
        self.ref = reference_actor
        self.critic = critic
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        world_model.eval(); reference_actor.eval(); critic.eval()
        self.reset()

    def reset(self):
        self.state = self.wm.rssm.initial_state(1, self.device)
        self.prev_action = torch.zeros(1, self.cfg.action_dim, device=self.device)
        self._last_A = {"survival": 1.0, "pred_return": 0.0, "risk": False}            # forecast cache
        self.traces = {k: [] for k in
                       ("survival", "pred_return", "risk", "d_steer", "d_throttle", "surprise", "value")}

    def step(self, obs, action, forecast=True):
        """`forecast=False` skips the expensive safety imagine and reuses the last forecast (for the
        live HUD on a weak laptop -- see should_forecast); the state-carry and the cheap style/value
        signals still run every step, so the recurrent state and the trace rows stay correct."""
        with torch.no_grad():
            e = self.wm.encoder(torch.as_tensor(obs, device=self.device).float().unsqueeze(0))
            self.state, _, _ = self.wm.rssm.obs_step(self.state, self.prev_action, e)
            feat = torch.cat(self.state, dim=-1)                                      # [h; z]
            if forecast:
                self._last_A = forecast_safety(self.wm, self.state, action, self.cfg.forecast_horizon)
            A = self._last_A
            B = style_deviation(self.ref, feat, action)
            C = state_value(self.critic, feat)
            self.prev_action = torch.as_tensor(np.asarray(action, np.float32), device=self.device).unsqueeze(0)

        out = {**A, **B, "value": C}
        for k in self.traces:
            self.traces[k].append(out[k])
        return out

    def finalize(self):
        return report_from_traces(self.traces, self.cfg)


def report_from_traces(traces, cfg):
    """PURE fusion: turn per-step signal traces into a session report -- summary stats + a list of
    discrete, labeled events (the actionable feedback). This is where 'feedback on driving habits'
    comes from: aggregate the events to see patterns ('oversteers on right curves', etc.)."""
    n = len(traces["survival"])
    events = []
    for t in range(n):
        if traces["risk"][t] or traces["survival"][t] < cfg.risk_threshold:
            events.append({"type": "near_off_road", "step": t, "survival": traces["survival"][t]})
        ds = traces["d_steer"][t]
        if abs(ds) > _STEER_DEV:
            events.append({"type": "oversteer", "step": t, "side": "left" if ds < 0 else "right", "d_steer": ds})
        if abs(traces["d_throttle"][t]) > _THROTTLE_DEV:
            kind = "harsh_throttle" if traces["d_throttle"][t] > 0 else "late_throttle"
            events.append({"type": kind, "step": t, "d_throttle": traces["d_throttle"][t]})

    def _mean(xs):
        return float(np.mean(xs)) if xs else 0.0

    counts = {}
    for e in events:
        counts[e["type"]] = counts.get(e["type"], 0) + 1
    return {
        "n_steps": n,
        "n_risk": int(sum(bool(r) for r in traces["risk"])),
        "mean_survival": _mean(traces["survival"]),
        "mean_value": _mean(traces["value"]),
        "mean_abs_steer_dev": _mean([abs(x) for x in traces["d_steer"]]),
        "mean_surprise": _mean(traces.get("surprise", [])),
        "event_counts": counts,
        "events": events,
    }
