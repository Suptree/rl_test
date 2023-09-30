# File: environment_manager.py

import torch
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs


class EnvironmentManager:
    def __init__(self, device, frame_skip):
        self.device = device
        self.frame_skip = frame_skip
        self.base_env = GymEnv(
            "InvertedDoublePendulum-v4", device=device, frame_skip=frame_skip
        )

    def setup_environment(self):
        env = TransformedEnv(
            self.base_env,
            Compose(
                ObservationNorm(in_keys=["observation"]),
                DoubleToFloat(in_keys=["observation"]),
                StepCounter(),
            ),
        )
        env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
        return env
