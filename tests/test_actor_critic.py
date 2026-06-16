"""Actor / Critic contracts + lambda-return math (spec §4.5, §4.6, §7)."""
import torch

from config import get_config
from models.actor_critic import Actor, Critic
from training.train_behavior import lambda_returns


def _cfg(**ov):
    d = dict(hidden_dim=32, min_std=0.1, action_dim=2)
    d.update(ov)
    return get_config(**d)


def test_actor_output_in_range_and_shapes():
    """Tanh-Normal actor: actions land in [-1,1]^A, plus a per-sample entropy estimate."""
    cfg = _cfg()
    feat_dim, A = 48, 2
    actor = Actor(cfg, feat_dim, A)
    feat = torch.randn(7, feat_dim)

    action, entropy = actor(feat)
    assert action.shape == (7, A)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)
    assert entropy.shape == (7,)


def test_actor_deterministic_is_reproducible():
    cfg = _cfg()
    actor = Actor(cfg, 48, 2)
    feat = torch.randn(5, 48)
    a1, _ = actor(feat, deterministic=True)
    a2, _ = actor(feat, deterministic=True)
    assert torch.equal(a1, a2)


def test_critic_output_shape():
    cfg = _cfg()
    critic = Critic(cfg, 48)
    v = critic(torch.randn(7, 48))
    assert v.shape == (7,)


def test_actor_log_prob_peaks_at_the_mode():
    """Tanh-Normal log-prob (with the change-of-variables correction): the mode (tanh(mean)) is
    more likely than an extreme action. Used for the 'surprise' style signal in eval/feedback.py."""
    torch.manual_seed(0)
    cfg = _cfg()
    actor = Actor(cfg, 48, 2)
    feat = torch.randn(5, 48)
    mode, _ = actor(feat, deterministic=True)

    lp_mode = actor.log_prob(feat, mode)
    lp_extreme = actor.log_prob(feat, torch.full_like(mode, 0.99))
    assert lp_mode.shape == (5,)
    assert torch.all(lp_mode > lp_extreme)        # the mode is the most likely action


def test_lambda_returns_matches_hand_worked_example():
    """V^lambda_t = r_t + gamma*c_t*[(1-lam)*v_{t+1} + lam*V^lambda_{t+1}],  V^lambda_H = v_H.

    Hand-worked (gamma=0.9, lam=0.5, H=2): rewards=[1,2], conts=[1,1], values=[.5,1,3].
      V_2 = v_2 = 3.0
      V_1 = 2 + 0.9*[0.5*3.0 + 0.5*3.0]              = 2 + 2.70  = 4.70
      V_0 = 1 + 0.9*[0.5*1.0 + 0.5*4.70]             = 1 + 2.565 = 3.565
    """
    rewards = torch.tensor([[1.0, 2.0]])
    conts = torch.tensor([[1.0, 1.0]])
    values = torch.tensor([[0.5, 1.0, 3.0]])

    out = lambda_returns(rewards, conts, values, gamma=0.9, lam=0.5)

    assert out.shape == (1, 2)
    assert torch.allclose(out, torch.tensor([[3.565, 4.70]]), atol=1e-4), out
