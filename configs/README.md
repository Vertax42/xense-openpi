# `configs/` — Per-task YAML training configs

This directory holds **one YAML per training/inference task**. Each file is a
self-contained representation of a `TrainConfig` and can be loaded by name:

```bash
# YAML file: configs/_examples/pi05_base_bi_flexiv_shoe_insole_..._h100.yaml
uv run scripts/train.py pi05_base_bi_flexiv_shoe_insole_..._h100 --exp-name=0519
```

## What's where

```
configs/
├── README.md                  ← this file
├── _examples/                 ← shared, in git, treat as templates
│   ├── debug_pi05.yaml
│   └── pi05_base_bi_flexiv_*.yaml
└── <your_task>.yaml           ← per-user (gitignored), put your in-flight configs here
```

The repo's `.gitignore` keeps **only** `configs/_examples/` (and this README)
under version control. Anything you drop into `configs/` directly is private
to your machine — perfect for in-flight experiments that don't need to be
shared yet.

## Lookup order (`get_config(name)`)

1. `configs/<name>.yaml`             — your local copy
2. `configs/_examples/<name>.yaml`   — shared example
3. Legacy `_CONFIGS` list in `src/openpi/training/config.py` — old configs that
   use non-YAML-friendly features (lambdas, `flax.nnx` filters, etc.)

## Writing a new config

Copy the closest example and edit. The bare minimum:

```yaml
# configs/_examples/<task_name>.yaml  (or configs/<task_name>.yaml for personal)

model:
  type: Pi0Config              # see openpi.training.registry.MODELS
  pi05: true
  paligemma_variant: gemma_2b
  action_expert_variant: gemma_300m

data:
  type: LeRobotBiFlexivDataConfig   # see openpi.training.registry.DATA_CONFIGS
  repo_id: Xense/<your_dataset>
  default_prompt: "..."
  base_config:
    prompt_from_task: true

weight_loader:
  type: CheckpointWeightLoader
  params_path: gs://openpi-assets/checkpoints/pi05_base/params

batch_size: 256
num_train_steps: 40000
num_workers: 64
fsdp_devices: 8
```

The **filename** is the config name — there is no `name:` field inside the
YAML. `exp_name` is supplied on the CLI (`--exp-name=...`).

## Full reference: `_FULL_REFERENCE.yaml`

[`configs/_examples/_FULL_REFERENCE.yaml`](_examples/_FULL_REFERENCE.yaml) lists
**every** `TrainConfig` field, every registered class for each polymorphic
slot, and the current default for each value. Open it alongside the example
you're editing — it answers "what does field X do" and "what other classes
can go in `data.type`" without forcing you to read 919 lines of `config.py`.

That file's name starts with `_` on purpose: the test suite skips it for the
equivalence check (it has no `_CONFIGS` counterpart), but a separate test
ensures it always parses cleanly so the docs can't silently rot.

## ⚠️ Before committing a YAML to `_examples/`

Read it once. **Do not check in machine-local absolute paths** such as
`/home/<you>/.../checkpoints/...`. Use the upstream URL
(`gs://openpi-assets/...`) or a path that works for every contributor.

## Registering a new class

Adding a new model/data-config/weight-loader Python class? Register its
string name in `src/openpi/training/registry.py` so YAML files can
reference it via `type: <YourClass>`.

## Regenerating examples after a config.py change

```bash
python scripts/migrate_configs_to_yaml.py --overwrite
```

This re-dumps everything in `_CONFIGS` that can be YAML-ified into
`configs/_examples/`. Run the equivalence test afterwards to verify:

```bash
pytest src/openpi/training/yaml_examples_equivalence_test.py
```

## Why are some configs NOT in `_examples/`?

A handful of legacy `_CONFIGS` entries use features that don't round-trip
through YAML — lambdas in `SimpleDataConfig`, `flax.nnx.All(...)` filters
for LoRA freezing, tokenizer **classes** (not instances) for some FAST/VQ
variants, or RoboArena's dynamically generated configs. Those stay in
`src/openpi/training/config.py::_CONFIGS` and are loaded via the
`_CONFIGS_DICT` fallback in `get_config()`.
