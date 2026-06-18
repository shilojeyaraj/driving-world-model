# Running on Kaggle (free GPU)

No local GPU? Kaggle gives ~30 free GPU-hours/week (T4 ×2 or P100). But be clear about **what
actually runs on Kaggle's headless Linux box**:

| Task | On Kaggle |
|---|---|
| **Image-mode world model on the visual DummyEnv** (dream video, drive-from-pixels) | ✅ pure torch+numpy, no display — **the clean GPU win** |
| State-mode toy training | ✅ runs, but it's CPU-fast already (the GPU barely helps the per-step `seq_len` loop) |
| **MetaDrive image** mode | ⚠️ needs offscreen OpenGL → run under `xvfb` (finicky; cells below) |
| MetaDrive state mode | ✅ runs, but it's CPU physics — GPU doesn't speed the sim |
| **DonkeyGym (Unity)** | ❌ a Unity GUI binary — not practical on Kaggle's headless box; do that locally |

**Bottom line:** use Kaggle for **GPU-accelerated image-mode training on the visual toy env** —
crisp 64×64 dream video and faster drive-from-pixels. The Unity/3-D sims want a local GPU+display.

## Step by step

**0. Push your code to GitHub** (Kaggle clones from there):
```bash
git push          # you currently have unpushed commits
```

**1.** kaggle.com → **New Notebook** → right panel → **Accelerator: GPU T4 ×2** (or P100), and
**Internet: On**.

**2. Get the code** (torch + CUDA + numpy are preinstalled on Kaggle):
```python
!git clone https://github.com/shilojeyaraj/driving-world-model.git
%cd driving-world-model
!pip -q install imageio
import torch; print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

**3. The GPU win — a crisp 64×64 dream video on the visual toy env:**
```python
from scripts.dream_video import dream_video
dream_video(image_size=64, wm_steps=3000, context=5, horizon=20, device="cuda")
# -> runs/dream/dream.gif  +  runs/dream/dream_montage.png
```
(On CPU this is ~30+ min at 64×64; on a T4 it's a few minutes.)

**4. View / download:**
```python
from IPython.display import Image
Image("runs/dream/dream.gif")
```
Outputs also appear under the notebook's **Output** tab — download `runs/dream/*` before the
session ends (Kaggle sessions are ephemeral).

Also GPU-ready (they respect `device="cuda"`):
```python
from config import get_config
from training.train_world_model import train
train(get_config(env="dummy", obs_type="image", image_size=64, device="cuda",
                 deter_dim=128, stoch_dim=32, hidden_dim=128, seq_len=20), steps=3000)
```
and `training.train_behavior` likewise.

## MetaDrive image on Kaggle (optional, finicky)
Headless OpenGL needs a virtual display:
```python
!apt-get -qq install -y xvfb
!pip -q install "git+https://github.com/metadriverse/metadrive.git"
!xvfb-run -a python -m scripts.probe_metadrive state
```
Image-observation rendering on a headless server is version-sensitive; if it errors, stick to the
visual DummyEnv for GPU image work.

## Notes
- Re-run `git push` locally after code changes, then re-clone (or `!git pull`) on Kaggle.
- DonkeyGym/Unity: run locally when you have a GPU + display (see `docs/DONKEYCAR.md`).
