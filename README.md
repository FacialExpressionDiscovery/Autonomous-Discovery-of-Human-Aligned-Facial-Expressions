# Autonomous Discovery of Human-Aligned Facial Expressions

Reinforcement learning and non-RL search baselines for discovering facial
expression parameters that drive a Furhat robot's rendered face toward a
target emotion, as scored by a facial expression recognition (FER) classifier.

**Project page:** a companion video and expression-performance demos are in
[`supplementary_videos/`](supplementary_videos/) and rendered at
[`index.html`](index.html) — see [Project page](#project-page) below for how
to view/host it.

Three interchangeable search strategies are provided, all optimizing the same
objective (a DAN classifier's probability of a target emotion on the rendered
Furhat face):

- **REINFORCE / Actor-Critic policy gradient** (`policy_gradient/`) — the main RL approach.
- **Random search** (`baselines/run_baseline.py`) — samples the 26-D action-unit vector uniformly at random.
- **Bayesian Optimization** (`baselines/run_bayesian.py`) — a Gaussian-process/UCB baseline built on a vendored copy of [`bayesian-optimization`](https://github.com/fmfn/BayesianOptimization) (see `baselines/vendor/`).

## Repository layout

```
config.yaml                    # REINFORCE training config
policy_gradient_agent.py       # REINFORCE entry point (backward-compatible shim)
policy_gradient/               # REINFORCE/Actor-Critic trainer, env, models
dan_classifier.py              # FER classifier: screen-captures the Furhat face, runs DAN
DAN/                           # Distract Your Attention Network (FER backbone)
baselines/
  run_baseline.py              # Random search entry point
  config_baseline.yaml
  run_bayesian.py              # Bayesian Optimization entry point
  config_bayesian.yaml
  bayesian_method.py           # BO baseline logic (built on the vendored engine)
  methods.py                   # Random search sampling logic
  common.py                    # Shared projection/evaluation/state utilities
  vendor/                      # Vendored bayesian-optimization package
tests/                         # pytest suite (Bayesian baseline)
furhat_template.png            # Screen-matching template (default character)
furhat_template_maurice.png    # Screen-matching template (Maurice character)
index.html, assets/            # Project page (see "Project page" below)
supplementary_videos/          # Companion video + expression-performance demos, with .vtt captions
```

## Requirements

- Python 3.10+
- A running **Furhat simulator** (the Furhat Virtual Robot app, part of the Furhat SDK) or a physical Furhat robot reachable on the network
- `pip install -r requirements.txt`
- DAN checkpoint(s) — download following the links in `DAN/README.md` and place under `DAN/models/`; referenced by `FER_MODEL_PATH` / `fer_model_path` in the config files. Checkpoints are not tracked in this repo (see `.gitignore`).

## How the classifier "sees" the robot's face

There is no direct image feed from Furhat — the code takes a **screenshot of
the simulator window** and locates the face by template-matching against
`furhat_template.png` (or `furhat_template_maurice.png` for the Maurice
character). This means the simulator window must actually be visible on
screen while a run is in progress.

Two capture modes are supported (`CAPTURE_FROM_MONITOR` in `config.yaml`,
`capture_from_monitor` in the baseline configs):

- **Single monitor** (`false`, default in the baseline configs): captures the
  primary monitor with `pyautogui.screenshot()`. Just keep the simulator
  window visible/foregrounded on your one screen.
- **Dual monitor** (`true`): captures a fixed region — `{left: 1920, top: 0,
  width: 1920, height: 1080}` — i.e. **a second 1920×1080 monitor placed to
  the right of a first 1920×1080 monitor**. This is the recommended setup for
  running unattended training/search:

  1. Connect two monitors, both at **1920×1080**, arranged left/right in your
     OS display settings (left monitor at x=0, right monitor starting at
     x=1920 — this is the Windows/OS default when you drag the second display
     to the right).
  2. **Right monitor:** open the Furhat simulator and make it fill this
     screen (fullscreen or maximized), so the robot's face is fully visible.
  3. **Left monitor:** open VS Code (or your terminal) here, and launch the
     script from this screen (`python policy_gradient_agent.py`,
     `python baselines/run_baseline.py`, or `python baselines/run_bayesian.py`).
  4. Do not cover the right monitor with other windows while a run is in
     progress — every evaluation is a live screenshot.

  If your monitor resolutions/arrangement differ, adjust the `monitor` dict
  in `dan_classifier.py` (`capture_face_image_right_screen` /
  `capture_virtual_furhat_face_right_screen`) accordingly.

## Connecting to Furhat

Set `FURHAT_IP` (`config.yaml`) / `furhat_ip` (baseline configs) to your
simulator's/robot's address — `127.0.0.1` if the simulator runs on the same
machine. The code talks to it via `furhat-remote-api`; optional idle-behavior
and head-pose control (`FURHAT_SET_HEAD_POSE`, `FURHAT_DISABLE_*`) additionally
use `furhat-realtime-api` if installed, and are skipped gracefully otherwise.

## Running

All three entry points read a YAML config next to them — no CLI flags needed
beyond an optional `--config` for the REINFORCE runner.

**REINFORCE / Actor-Critic** (edit `config.yaml`, then):
```
python policy_gradient_agent.py --config config.yaml
```
Outputs land under `output_REINFORCE/<job_id>/`.

**Random search baseline** (edit `baselines/config_baseline.yaml`, then, from the repo root):
```
python baselines/run_baseline.py
```
Outputs land under `output_baseline/<job_id>/`.

**Bayesian Optimization baseline** (edit `baselines/config_bayesian.yaml`, then, from the repo root):
```
python baselines/run_bayesian.py
```
Outputs land under `output_baseline_bo/<job_id>/`.

Each run directory contains a full evaluation log (CSV), the top-N configs
and annotated face images, `configs.npy` / `rewards.npy`, `metadata.json`,
and a copy of the config used — see the docstring at the top of each
`run_*.py` for the exact file list.

## Tests

```
pytest tests/
```
Covers the Bayesian Optimization baseline end-to-end with all hardware
(Furhat, DAN) mocked — no simulator or GPU required.

## Project page

`index.html` (with `assets/` and `supplementary_videos/`) is a static,
dependency-free project page: an overview of the method, the companion video,
and the two silent expression-performance demos, each with `.vtt` captions.

To view it locally, just open `index.html` in a browser — no build step.

To host it with **GitHub Pages** once this repository is pushed:

1. Push this folder as the repository root (the `index.html` at the top level
   is what Pages will serve).
2. In the repository, go to **Settings → Pages**.
3. Under **Build and deployment → Source**, choose **Deploy from a branch**.
4. Pick the branch (e.g. `main`) and folder **`/ (root)`**, then **Save**.
5. GitHub publishes the page at `https://<account>.github.io/<repo>/` within
   a few minutes (the exact URL is shown on the same Settings → Pages screen).

A `.nojekyll` file is included so GitHub Pages serves the site as-is,
without running it through Jekyll.
