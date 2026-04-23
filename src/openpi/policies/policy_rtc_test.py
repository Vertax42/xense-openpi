import numpy as np

from openpi.models import model as _model
from openpi.policies import bi_flexiv_policy
from openpi.policies import policy as _policy
from openpi import transforms as _transforms


class _CaptureModel(_model.BaseModel):
    def __init__(self, action_dim: int = 20, action_horizon: int = 4):
        super().__init__(action_dim=action_dim, action_horizon=action_horizon, max_token_len=16)
        self.sample_kwargs = None

    def compute_loss(self, rng, observation, actions, *, train: bool = False):
        raise NotImplementedError

    def sample_actions(self, rng, observation, **kwargs):
        self.sample_kwargs = kwargs
        batch = observation.state.shape[0]
        return np.zeros((batch, self.action_horizon, self.action_dim), dtype=np.float32)


def test_rtc_prev_chunk_left_over_uses_training_action_space():
    model = _CaptureModel()
    delta_mask = _transforms.make_bool_mask(18, -1, -1)
    norm_stats = {
        "state": _transforms.NormStats(
            mean=np.zeros(20, dtype=np.float32),
            std=np.ones(20, dtype=np.float32),
            q01=np.zeros(20, dtype=np.float32),
            q99=np.ones(20, dtype=np.float32),
        ),
        "actions": _transforms.NormStats(
            mean=np.full(20, 10.0, dtype=np.float32),
            std=np.full(20, 2.0, dtype=np.float32),
            q01=np.zeros(20, dtype=np.float32),
            q99=np.ones(20, dtype=np.float32),
        ),
    }
    policy = _policy.Policy(
        model,
        transforms=[
            bi_flexiv_policy.BiFlexivInputs(),
            _transforms.DeltaActions(delta_mask),
            _transforms.Normalize(norm_stats),
        ],
    )

    obs = {
        "state": np.arange(20, dtype=np.float32),
        "images": {
            "head": np.zeros((3, 4, 4), dtype=np.uint8),
            "left_wrist": np.zeros((3, 4, 4), dtype=np.uint8),
            "right_wrist": np.zeros((3, 4, 4), dtype=np.uint8),
        },
        "prompt": "test",
    }
    prev_chunk_abs = np.stack(
        [
            np.arange(20, dtype=np.float32) + 100.0,
            np.arange(20, dtype=np.float32) + 200.0,
        ],
        axis=0,
    )

    policy.infer(obs, prev_chunk_left_over=prev_chunk_abs, inference_delay=2)

    transformed = model.sample_kwargs["prev_chunk_left_over"]
    assert transformed.shape == (1, 2, 20)

    expected = prev_chunk_abs.copy()
    expected[:, :18] -= obs["state"][:18]
    expected = (expected - 10.0) / 2.000001

    np.testing.assert_allclose(np.asarray(transformed[0]), expected, rtol=1e-5, atol=1e-5)
