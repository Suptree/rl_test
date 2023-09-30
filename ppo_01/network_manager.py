import torch.nn as nn
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator


class NetworkManager:
    def __init__(self, num_cells, env, device):
        self.num_cells = num_cells
        self.env = env
        self.device = device

    def setup_networks(self):
        actor_net = nn.Sequential(
            nn.LazyLinear(self.num_cells, device=self.device),
            nn.Tanh(),
            nn.LazyLinear(self.num_cells, device=self.device),
            nn.Tanh(),
            nn.LazyLinear(self.num_cells, device=self.device),
            nn.Tanh(),
            nn.LazyLinear(2 * self.env.action_spec.shape[-1], device=self.device),
            NormalParamExtractor(),
        )

        policy_module = TensorDictModule(
            actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
        )

        policy_module = ProbabilisticActor(
            module=policy_module,
            spec=self.env.action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "min": self.env.action_spec.space.minimum,
                "max": self.env.action_spec.space.maximum,
            },
            return_log_prob=True,
        )

        value_net = nn.Sequential(
            nn.LazyLinear(self.num_cells, device=self.device),
            nn.Tanh(),
            nn.LazyLinear(self.num_cells, device=self.device),
            nn.Tanh(),
            nn.LazyLinear(self.num_cells, device=self.device),
            nn.Tanh(),
            nn.LazyLinear(1, device=self.device),
        )

        value_module = ValueOperator(module=value_net, in_keys=["observation"])

        return policy_module, value_module
