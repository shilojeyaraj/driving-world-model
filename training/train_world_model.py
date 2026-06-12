"""World-model training loop skeleton. The loop is here; the learning signal comes from
WorldModel.assemble_loss(), which YOU implement (models/world_model.py).

OK TO USE A LIBRARY for the loop; the LOSS is yours.   Run:  python -m training.train_world_model
"""
import torch

from config import get_config
from training.collect import collect


def train(cfg, steps=10_000):
    from models.world_model import WorldModel

    device = torch.device(cfg.device)
    buf = collect(cfg, num_steps=5000)
    model = WorldModel(cfg, cfg.action_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for step in range(steps):
        if not buf.can_sample():
            break
        batch = {k: torch.as_tensor(v, device=device) for k, v in buf.sample(cfg.batch_size).items()}
        loss, metrics = model.assemble_loss(batch)        # <- you implement this
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)   # why clip? see CONCEPTS.md
        opt.step()
        if step % 100 == 0:
            print(step, {k: round(float(v), 4) for k, v in metrics.items()})


if __name__ == "__main__":
    # state-obs run -> fine on a laptop CPU. For images: obs_type="image", device="cuda" on Kaggle.
    train(get_config(env="dummy", obs_type="state", device="cpu", max_episode_steps=200))
