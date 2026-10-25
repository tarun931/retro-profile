# Training notes

## Recommended hardware

  * Single GPU with ≥ 24 GB VRAM (e.g. RTX 3090, A5000) trains the full
    RETRO-Profile on TIMIT in ~3 h.
  * On A100-40 GB you can raise `data.batch_size` to 32 and `data.bag_size`
    to 16 for slightly better numbers.
  * CPU-only is supported but very slow; useful only for tests.

## Memory tips

  * The base WavLM encoder is **frozen by default** (`model.encoder.freeze:
    true`). Unfreezing it ~triples memory and gives only a marginal win
    in our experience — try this last.
  * `precision: 16-mixed` is the default and works fine on Ampere+ GPUs.
  * If you OOM on A100, reduce `data.bag_size` first, then `data.batch_size`.

## Hyperparameters worth touching

| Knob | Default | Notes |
|---|---|---|
| `model.pooling.num_heads` | 4 | More heads help on bag sizes ≥ 8 |
| `model.heads.height.delta_clip` | 5 | Larger = more freedom to deviate from kNN anchor |
| `training.loss.ordcon_weight` | 0.3 | Too high suppresses regression; too low makes kNN noisy |
| `data.bag_size` | 8 | TIMIT has 10 utts/speaker; use 8 for stochastic bagging |
| `data.augmentation.speed_perturb` | [0.95, 1.05] | Wider hurts height regression |

## Curriculum

For best results, run a small **two-phase schedule**:

  1. **Phase 1 (epochs 0-30)**: train without retrieval (`use_retrieval: false`
     on the heads, or `experiment=no_knn`). The OrdCon loss organizes the
     embedding space.
  2. **Phase 2 (epochs 30-60)**: build the datastore from the phase-1
     checkpoint and resume with retrieval enabled. The residual heads now
     learn to correct kNN anchors instead of regressing from scratch.

We provide a single `experiment=retro_profile_full` recipe for simplicity,
which trains both phases in one go and relies on the empty-datastore
fallback for the early epochs. The two-phase recipe is slightly stronger.

## Logging

  * TensorBoard is enabled by default. Run `tensorboard --logdir runs/`.
  * W&B is supported — set `logger.wandb.enabled: true` in the config or
    `+logger.wandb.enabled=true` on the CLI.

## Reproducibility

  * All scripts seed Python, NumPy, and PyTorch (CPU + CUDA).
  * `seed_everything(seed, deterministic=True)` further forces cuDNN
    determinism — but is much slower; not enabled by default.
  * Use `requirements.txt` (pinned versions) for bit-stable repros.
