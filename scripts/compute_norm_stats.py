"""Compute normalization statistics for a config.

For LeRobot datasets the script reads parquet files **directly** via pyarrow,
skipping the LeRobot dataset abstraction and all image decoding entirely.
Only the state / action columns are loaded, action-horizon sequences are
assembled per episode in numpy, and the relevant transforms (e.g. DeltaActions)
are applied before feeding into the running statistics accumulator.

This is typically 10-30× faster than going through the full data-loader pipeline.
"""

import logging
import pathlib

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms

logger = logging.getLogger(__name__)

# Only these transform types touch state / actions without needing images.
# They must be applied even in the fast parquet path.
_STATE_ACTION_TRANSFORM_TYPES = (
    transforms.DeltaActions,
    transforms.SubsampleActions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_parquet_key(repack_inputs: list, dest_key: str) -> str | None:
    """Return the parquet column name mapped to *dest_key* by RepackTransform."""
    for t in repack_inputs:
        if isinstance(t, transforms.RepackTransform):
            structure = t.structure
            if isinstance(structure, dict):
                val = structure.get(dest_key)
                if isinstance(val, str):
                    return val
    return None


def _state_action_transforms(data_transforms_inputs: list) -> list:
    """Return transforms that operate purely on state / actions (no images)."""
    return [t for t in data_transforms_inputs if isinstance(t, _STATE_ACTION_TRANSFORM_TYPES)]


# ---------------------------------------------------------------------------
# Fast parquet-based data loader (LeRobot datasets only)
# ---------------------------------------------------------------------------


def create_parquet_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int = 512,
    max_frames: int | None = None,
) -> tuple:
    """Yield {state, actions} batches by reading parquet files directly.

    No LeRobot dataset / torch DataLoader / image decoding overhead.

    Returns:
        (generator, num_batches)  –  num_batches is an *upper bound* used for
        tqdm; the generator may yield slightly fewer batches when max_frames is
        set or the last batch is partial.
    """
    import pyarrow.dataset as pad
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("data_config.repo_id must be set.")

    # --- resolve parquet column names from repack_transforms --------------------
    state_col = _extract_parquet_key(data_config.repack_transforms.inputs, "state")
    action_col = _extract_parquet_key(data_config.repack_transforms.inputs, "actions")
    if state_col is None or action_col is None:
        raise ValueError(
            f"Cannot infer parquet column names from repack_transforms "
            f"(state={state_col!r}, actions={action_col!r}). "
            "Ensure the config uses a standard RepackTransform."
        )
    logger.info("Parquet columns: state=%r, action=%r", state_col, action_col)

    # --- load parquet -----------------------------------------------------------
    meta = LeRobotDatasetMetadata(repo_id)
    data_dir = meta.root / "data"
    ds = pad.dataset(str(data_dir), format="parquet")

    cols = ["episode_index", "frame_index", state_col, action_col]
    table = ds.to_table(columns=cols)
    table = table.sort_by([("episode_index", "ascending"), ("frame_index", "ascending")])

    episode_arr = np.array(table.column("episode_index").to_pylist())
    states_all = np.array(table.column(state_col).to_pylist(), dtype=np.float32)   # (N, sd)
    actions_all = np.array(table.column(action_col).to_pylist(), dtype=np.float32) # (N, ad)

    n_total = len(episode_arr)
    logger.info("Loaded %d frames from %s", n_total, data_dir)

    # --- episode boundary indices -----------------------------------------------
    ep_change = np.flatnonzero(np.diff(episode_arr, prepend=episode_arr[0] - 1))
    ep_starts = ep_change.tolist()
    ep_ends = ep_change[1:].tolist() + [n_total]

    # --- state/action transforms to apply ---------------------------------------
    sa_transforms = _state_action_transforms(data_config.data_transforms.inputs)

    # account for max_frames
    frames_limit = max_frames if max_frames is not None else n_total
    num_batches = min(frames_limit, n_total + batch_size - 1) // batch_size

    def _gen():
        buf_s: list[np.ndarray] = []
        buf_a: list[np.ndarray] = []
        frames_yielded = 0

        for ep_s, ep_e in zip(ep_starts, ep_ends):
            if frames_yielded >= frames_limit:
                break

            ep_len = ep_e - ep_s
            ep_states = states_all[ep_s:ep_e]        # (ep_len, sd)
            ep_actions_flat = actions_all[ep_s:ep_e]  # (ep_len, ad)

            # build action-horizon sequences with end-of-episode clamping
            row_idx = np.arange(ep_len)[:, None] + np.arange(action_horizon)[None, :]
            row_idx = np.clip(row_idx, 0, ep_len - 1)   # (ep_len, ah)
            ep_action_seqs = ep_actions_flat[row_idx]    # (ep_len, ah, ad)

            buf_s.append(ep_states)
            buf_a.append(ep_action_seqs)

            # flush complete batches
            while sum(len(x) for x in buf_s) >= batch_size:
                s = np.concatenate(buf_s, axis=0)
                a = np.concatenate(buf_a, axis=0)

                batch = {"state": s[:batch_size].copy(), "actions": a[:batch_size].copy()}
                for t in sa_transforms:
                    batch = t(batch)
                yield batch
                frames_yielded += batch_size

                rem_s, rem_a = s[batch_size:], a[batch_size:]
                buf_s = [rem_s] if len(rem_s) else []
                buf_a = [rem_a] if len(rem_a) else []

                if frames_yielded >= frames_limit:
                    return

        # final partial batch
        if buf_s and frames_yielded < frames_limit:
            s = np.concatenate(buf_s, axis=0)
            a = np.concatenate(buf_a, axis=0)
            if len(s):
                batch = {"state": s.copy(), "actions": a.copy()}
                for t in sa_transforms:
                    batch = t(batch)
                yield batch

    return _gen(), num_batches


# ---------------------------------------------------------------------------
# Original LeRobot torch-based data loader (kept as fallback / RLDS path)
# ---------------------------------------------------------------------------


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.DataLoader, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(dataset, num_batches=num_batches)
    return data_loader, num_batches


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(
    config_name: str,
    max_frames: int | None = None,
    batch_size: int = 512,
    use_torch_loader: bool = False,
):
    """Compute normalization statistics for a config.

    Args:
        config_name: Name of the training config.
        max_frames: Optional cap on the number of frames to process.
        batch_size: Batch size for the parquet reader (default 512).
        use_torch_loader: Fall back to the original LeRobot torch DataLoader
            (slower, decodes images via the dataset pipeline).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        # RLDS path (DROID etc.) – unchanged
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    elif use_torch_loader:
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers,
            max_frames,
        )
    else:
        # Fast parquet path (default for LeRobot datasets)
        data_loader, num_batches = create_parquet_dataloader(
            data_config,
            config.model.action_horizon,
            batch_size=batch_size,
            max_frames=max_frames,
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: s.get_statistics() for key, s in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
