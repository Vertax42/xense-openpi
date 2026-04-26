# openpi - Xense Robotics Fork

> Fork of [Physical Intelligence's openpi](https://github.com/Physical-Intelligence/openpi),
> adapted and extended for **Xense Robotics platforms** (BiARX5, Xense Flare, BiFlexiv Rizon4).

## 🎯 Our Contributions

- **Xense platform support** — BiARX5, Xense Flare, BiFlexiv Rizon4 dual-arm setups
- **Custom training configs** — tie shoes, pick-and-place, open lock, wipe vase, assemble box, …
- **Platform-specific policies** — `xense_flare_policy.py`, `bi_flexiv_*` runtimes, optimized data pipelines
- **Real-time control** — RTC (Real-Time Chunking) inference + training-time RTC, both JAX and PyTorch
- **Streamlined codebase** — dropped ALOHA / LIBERO; focus on DROID and Xense platforms

For everything else (model card, generic fine-tuning workflow, full checkpoint catalogue), see
the [upstream openpi README](https://github.com/Physical-Intelligence/openpi#readme).

## Requirements

**⚠️ NVIDIA GPU with CUDA 12 required. CPU-only execution is not supported.**

| Mode               | Min VRAM | Recommended GPU    |
| ------------------ | -------- | ------------------ |
| Inference          | 24 GB    | RTX 4090 (24 GB)   |
| Fine-tuning (LoRA) | 24 GB    | RTX 4090 (24 GB)   |
| Fine-tuning (full) | 70+ GB   | A100 (80 GB) / H100 |

Multi-GPU FSDP is supported (`fsdp_devices` in the training config). Multi-node is not.
OS: Ubuntu 22.04+ (other Linux distros may work but are not tested).

## Installation

We build on the `lerobot-xense-py312` mamba environment. Inside that env:

```bash
# Clone with submodules
git clone git@github.com:Vertax42/xense-openpi.git openpi
cd openpi

mamba activate lerobot-xense-py312

# 1. Install the client package (xense-client)
GIT_LFS_SKIP_SMUDGE=1 pip install -e packages/xense-client

# 2. Install the main openpi package
GIT_LFS_SKIP_SMUDGE=1 pip install -e .
```

> The repo previously shipped a `uv.lock`; we now manage dependencies through the mamba env
> rather than uv. The `[tool.uv.*]` sections in `pyproject.toml` are kept only for users who
> still want to install via uv.

### PyTorch path

`pip install -e .` is sufficient. As of `transformers==5.3.0`, Pi0-specific behaviour
(AdaRMS, activation precision, read-only KV cache) lives in
`src/openpi/models_pytorch/transformers_compat/` as small subclasses. The old
`transformers_replace/` + `cp` workflow has been removed.

## Model Checkpoints

Base models (`pi0_base`, `pi0_fast_base`, `pi05_base`) and DROID expert checkpoints
are downloaded from `gs://openpi-assets/checkpoints/...` on first use and cached under
`~/.cache/openpi` (override with `OPENPI_DATA_HOME`). Full catalogue and licensing
in the [upstream README](https://github.com/Physical-Intelligence/openpi#model-checkpoints).

## Inference & Fine-tuning Quickstart

### Run a pre-trained model

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")
policy = policy_config.create_trained_policy(config, checkpoint_dir)
action_chunk = policy.infer(example)["actions"]
```

A working notebook: [`examples/inference.ipynb`](examples/inference.ipynb).

### Fine-tune on your own data

The flow is upstream-standard — convert to LeRobot dataset, define a `DataConfig` /
`TrainConfig` (see `src/openpi/training/config.py`), compute norm stats, train, serve.
For a concrete walkthrough using DROID, see the
[upstream fine-tuning section](https://github.com/Physical-Intelligence/openpi#fine-tuning-base-models-on-your-own-data).
Xense-platform configs live alongside the existing ones and are exercised by the commands
in the next section.

### Remote inference

The `xense-client` package provides a websocket-based policy client, so model inference
can run on a remote GPU server while the robot runtime streams observations and consumes
action chunks in real time. Minimal reference: `examples/simple_client/`.

## PyTorch Support

PyTorch implementations of π₀ and π₀.₅ are validated on the DROID benchmark for
inference and fine-tuning. Currently unsupported in the PyTorch path:

- π₀-FAST
- Mixed precision training (full bf16 or fp32 only; controlled by `pytorch_training_precision`)
- FSDP, LoRA, EMA weights

Convert a JAX checkpoint to PyTorch:

```bash
python examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /path/to/jax/checkpoint \
    --config_name <config name> \
    --output_path /path/to/pytorch/checkpoint
```

Train (single GPU / DDP):

```bash
# Single GPU
python scripts/train_pytorch.py <config_name> --exp_name <run_name>

# Multi-GPU, single node
torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> \
    scripts/train_pytorch.py <config_name> --exp_name <run_name>

# Resume
python scripts/train_pytorch.py <config_name> --exp_name <run_name> --resume
```

The serving path is identical to JAX — `scripts/serve_policy.py` auto-detects the
PyTorch checkpoint format.

---

## 🤖 Xense Platform Training & Deployment

Production commands for training and deploying on Xense platforms.

### Platforms

- **BiARX5** — bi-manual ARX-5 with parallel grippers
- **Xense Flare** — UMI-style dual-arm with data-collection grippers
- **BiFlexiv** — dual-arm Flexiv Rizon4, real-time

### Optional environment variables

```bash
# Offline mode for locally-cached HuggingFace datasets
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# NCCL settings for multi-GPU on hosts without NVLink/IB
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
```

### Training (latest per platform)

#### BiARX5 — training-time RTC

```bash
python scripts/compute_norm_stats.py --config-name tie_shoes_50_episodes_no_adjust_training_time_rtc_0426_h100
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py \
    tie_shoes_50_episodes_no_adjust_training_time_rtc_0426_h100 \
    --exp-name=tie_shoes_50_episodes_no_adjust_training_time_rtc_0426_h100 --overwrite
```

#### BiFlexiv — assemble box with phone stand

```bash
python scripts/compute_norm_stats.py --config-name pi05_base_bi_flexiv_assemble_box_with_phone_stand_lora_0403
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py \
    pi05_base_bi_flexiv_assemble_box_with_phone_stand_lora_0403 \
    --exp-name=bi_flexiv_assemble_box_with_phone_stand_lora_20260403 --overwrite

python scripts/compute_norm_stats.py --config-name pi05_base_bi_flexiv_assemble_box_with_phone_stand_lora_0422_merged_fixed_h100
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py \
    pi05_base_bi_flexiv_assemble_box_with_phone_stand_lora_0422_merged_fixed_h100 \
    --exp-name=pi05_base_bi_flexiv_assemble_box_with_phone_stand_lora_0422_merged_fixed_h100_0422 --overwrite
```

### Deployment (latest per platform)

#### BiARX5 — training-time RTC inference

```bash
python scripts/serve_policy.py \
    --default-prompt="pick rgb cubes and place them into the blue box" \
    policy:checkpoint \
    --policy.config=pi05_base_arx5_lora_training_time_rtc \
    --policy.dir=checkpoints/pi05_base_arx5_lora_training_time_rtc/training_time_rtc_20251209/39999
```

#### Xense Flare — open lock inference

```bash
python scripts/serve_policy.py \
    --default-prompt="open the lock with the key" \
    policy:checkpoint \
    --policy.config=pi05_base_xense_flare_open_lock_rtc_0228 \
    --policy.dir=checkpoints/pi05_base_xense_flare_open_lock_rtc_0228/xense_flare_open_lock_rtc_0228/19999
```

#### BiFlexiv — assemble box inference

```bash
python scripts/serve_policy.py \
    --default-prompt="assemble the box with the phone stand" \
    policy:checkpoint \
    --policy.config=pi05_base_bi_flexiv_assemble_box_with_phone_stand_lora_0403 \
    --policy.dir=checkpoints/pi05_base_bi_flexiv_assemble_box_with_phone_stand_lora_0403/bi_flexiv_assemble_box_with_phone_stand_lora_20260403/19999

python scripts/serve_policy.py \
    --default-prompt="assemble the box with the phone stand" \
    policy:checkpoint \
    --policy.config=pi05_base_bi_flexiv_assemble_box_with_phone_stand_lora_0410_merged_fixed \
    --policy.dir=checkpoints/pi05_base_bi_flexiv_assemble_box_with_phone_stand_lora_0410_merged_fixed/bi_flexiv_assemble_box_with_phone_stand_lora_0410_merged_fixed_20260413/39000
```

### Robot client

```bash
# BiARX5 with tactile sensors
python -m examples.bi_arx5_real.main \
    --args.host 192.168.2.215 \
    --args.port 8000 \
    --args.dry_run \
    --args.enable_tactile_sensors

# BiFlexiv RT, side mount, RTC
python -m examples.bi_flexiv_rizon4_rt.main \
    --args.host 192.168.142.158 \
    --args.port 8000 \
    --args.bi-mount-type side \
    --args.inner-control-hz 1000 \
    --args.interpolate-cmds \
    --args.runtime-hz 30 \
    --args.rtc-enabled \
    --args.dry-run

# BiFlexiv RT, forward mount, RTC
python -m examples.bi_flexiv_rizon4_rt.main \
    --args.host 192.168.142.220 \
    --args.port 8000 \
    --args.bi-mount-type forward \
    --args.inner-control-hz 1000 \
    --args.interpolate-cmds \
    --args.runtime-hz 30 \
    --args.rtc-enabled \
    --args.dry-run
```

## Troubleshooting

| Issue | Resolution |
| --- | --- |
| Dependency conflicts | Make sure you're in the `lerobot-xense-py312` mamba env, then `GIT_LFS_SKIP_SMUDGE=1 pip install -e .`. |
| Training OOM | Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`; use `--fsdp-devices <n>` for FSDP; consider disabling EMA. |
| Policy server connection errors | Confirm the server is listening on the expected port; check firewall between client and server. |
| Missing norm stats | Run `python scripts/compute_norm_stats.py --config-name <config>` before training. |
| Diverging training loss | Inspect `q01`/`q99`/`std` in `norm_stats.json` — rarely-used dims can produce huge normalized values; manually clamp if needed. |
| Action dimension mismatch | Verify your data transforms match the robot's action space (see `xense_flare_policy.py`, `bi_flexiv_*` policies). |

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Acknowledgments

Based on the original [OpenPI](https://github.com/Physical-Intelligence/openpi) by
Physical Intelligence. Thanks to the PI team for open-sourcing this work.
