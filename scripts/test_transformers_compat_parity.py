"""Numerical parity check for the transformers==5.3.0 migration.

Strategy
--------
Random-init parity is unreliable across versions: the order of ``nn.Module``
construction changes the RNG consumption even when architectures are otherwise
equivalent. Instead this script transfers weights:

- ``capture`` (run on ``main`` / transformers==4.53.2 + transformers_replace)
  builds the model, runs the four Pi0 forward paths, and saves *both* the
  ``state_dict`` and the resulting outputs.
- ``compare`` (run on ``feat/transformers-5.3.0``) builds the Pi-subclassed
  model, loads the captured ``state_dict`` with ``strict=False`` (the new
  compat layer may have a handful of extra keys inherited from upstream), runs
  the same deterministic inputs through the same forward paths, and compares
  the outputs tensor-by-tensor.

Usage
-----
    # 1) Freeze baseline on main / patched 4.53.2
    uv run scripts/test_transformers_compat_parity.py capture \
        --output tests/fixtures/parity_golden_v4_53_2.pt

    # 2) Compare on feat/transformers-5.3.0
    uv run scripts/test_transformers_compat_parity.py compare \
        --golden tests/fixtures/parity_golden_v4_53_2.pt

``bfloat16`` weights land around 1e-3; ``float32`` (default) tolerates 1e-5.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import torch
import transformers

from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel

# Match real Pi0 enough to exercise the projector + text hidden-size alignment
# (both old main and new compat set ``vision_config.projection_dim`` equal to
# VLM width). 1024 keeps the model <5 GB so weights + activations fit alongside
# a SigLIP base tower on a 16 GB card.
VLM_WIDTH = 1024
EXPERT_WIDTH = 1024
DEPTH = 2


def build_model(precision: Literal["float32", "bfloat16"] = "float32", device: str = "cpu"):
    torch.manual_seed(0)
    shared = {"num_heads": 8, "head_dim": 32, "num_kv_heads": 1}
    vlm_cfg = SimpleNamespace(width=VLM_WIDTH, mlp_dim=VLM_WIDTH * 2, depth=DEPTH, **shared)
    expert_cfg = SimpleNamespace(width=EXPERT_WIDTH, mlp_dim=EXPERT_WIDTH * 2, depth=DEPTH, **shared)
    model = PaliGemmaWithExpertModel(vlm_cfg, expert_cfg, use_adarms=[False, True], precision=precision)
    return model.to(device).eval()


def make_inputs(device: str, seed: int = 42):
    torch.manual_seed(seed)
    batch, t_prefix, t_suffix = 1, 8, 4
    prefix = torch.randn(batch, t_prefix, VLM_WIDTH, device=device)
    suffix = torch.randn(batch, t_suffix, EXPERT_WIDTH, device=device)
    adarms = torch.randn(batch, EXPERT_WIDTH, device=device)
    return {
        "prefix": prefix,
        "suffix": suffix,
        "adarms": adarms,
        "t_prefix": t_prefix,
        "t_suffix": t_suffix,
        "image": torch.randn(1, 3, 224, 224, device=device),
    }


def run_forwards(model, inputs: dict, device: str) -> dict:
    prefix = inputs["prefix"].to(device)
    suffix = inputs["suffix"].to(device)
    adarms = inputs["adarms"].to(device)
    t_prefix = int(inputs["t_prefix"])
    t_suffix = int(inputs["t_suffix"])

    pos_p = torch.arange(t_prefix, device=device).unsqueeze(0)
    attn_p = torch.zeros(1, 1, t_prefix, t_prefix, device=device)
    prefix_out_list, pkv = model(
        inputs_embeds=[prefix, None],
        attention_mask=attn_p,
        position_ids=pos_p,
        use_cache=True,
        adarms_cond=[None, None],
    )

    pos_s = torch.arange(t_prefix, t_prefix + t_suffix, device=device).unsqueeze(0)
    attn_s = torch.zeros(1, 1, t_suffix, t_prefix + t_suffix, device=device)
    suffix_out_list, _ = model(
        inputs_embeds=[None, suffix],
        attention_mask=attn_s,
        position_ids=pos_s,
        past_key_values=pkv,
        use_cache=False,
        adarms_cond=[None, adarms],
    )

    pos_j = torch.arange(t_prefix + t_suffix, device=device).unsqueeze(0)
    attn_j = torch.zeros(1, 1, t_prefix + t_suffix, t_prefix + t_suffix, device=device)
    joint_out_list, _ = model(
        inputs_embeds=[prefix, suffix],
        attention_mask=attn_j,
        position_ids=pos_j,
        use_cache=False,
        adarms_cond=[None, adarms],
    )

    image_feat = model.embed_image(inputs["image"].to(device))

    return {
        "prefix_only": prefix_out_list[0].detach().cpu().float(),
        "suffix_only": suffix_out_list[1].detach().cpu().float(),
        "joint_prefix": joint_out_list[0].detach().cpu().float(),
        "joint_suffix": joint_out_list[1].detach().cpu().float(),
        "image_features": image_feat.detach().cpu().float(),
    }


def capture(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Build on CPU first so weights live there; move to GPU just for forward.
    model = build_model(precision=args.precision, device="cpu")
    state_dict_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model = model.to(device)
    inputs = make_inputs(device)
    outs = run_forwards(model, inputs, device)

    inputs_cpu = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    payload = {
        "transformers_version": transformers.__version__,
        "precision": args.precision,
        "state_dict": state_dict_cpu,
        "inputs": inputs_cpu,
        "outputs": outs,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(
        f"Saved baseline ({len(outs)} tensors + {len(state_dict_cpu)} state_dict keys, "
        f"transformers={transformers.__version__}) -> {out}"
    )


def compare(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(args.golden, map_location="cpu", weights_only=False)

    # Build and load on CPU, then move to GPU so we never hold 2x model weight
    # copies on the device.
    model = build_model(precision=payload["precision"], device="cpu")
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if missing:
        print(f"  missing keys ({len(missing)}): {missing[:3]}...")
    if unexpected:
        print(f"  unexpected keys ({len(unexpected)}): {unexpected[:3]}...")
    model = model.to(device)

    new_outs = run_forwards(model, payload["inputs"], device)

    print(f"baseline transformers=={payload['transformers_version']}, current=={transformers.__version__}")
    ok = True
    for k, golden in payload["outputs"].items():
        got = new_outs[k]
        if got.shape != golden.shape:
            print(f"  [FAIL] {k}: shape mismatch golden={tuple(golden.shape)} got={tuple(got.shape)}")
            ok = False
            continue
        diff = (got - golden).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        status = "OK" if max_abs <= args.atol else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  [{status}] {k}: max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}, shape={tuple(got.shape)}")
    raise SystemExit(0 if ok else 1)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    cap = sub.add_parser("capture", help="Save state_dict + inputs + outputs from the current transformers install.")
    cap.add_argument("--output", required=True)
    cap.add_argument("--precision", choices=["float32", "bfloat16"], default="float32")
    cap.set_defaults(func=capture)

    cmp = sub.add_parser("compare", help="Transfer captured weights into the compat model and compare outputs.")
    cmp.add_argument("--golden", required=True)
    cmp.add_argument("--atol", type=float, default=1e-5)
    cmp.set_defaults(func=compare)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
