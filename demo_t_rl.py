import torch
from tensordict import TensorDict
from torchrl.data import PrioritizedReplayBuffer, ReplayBuffer
batch_size = 5
tensordict = TensorDict(
    source={
        "key 1": torch.zeros(batch_size, 3),
        "key 2": torch.zeros(batch_size, 5, 6, dtype=torch.bool),
    },
    batch_size=[batch_size],
)
# print(tensordict)
# rb = ReplayBuffer(collate_fn=lambda x: x)
# print(rb.add(1))
# print(rb.sample(1))
# rb.extend([2, 3])
# print(rb.sample(3))




# collate_fn = torch.stack
# rb = ReplayBuffer(collate_fn=collate_fn)
# rb.add(TensorDict({"a": torch.randn(3)}, batch_size=[]))
# print(len(rb))

# rb.extend(TensorDict({"a": torch.randn(2, 3)}, batch_size=[2]))
# print(len(rb))
# # print(rb.sample(1))
# print(rb.sample(2).contiguous())


# torch.manual_seed(0)
# from torchrl.data import TensorDictPrioritizedReplayBuffer

# rb = TensorDictPrioritizedReplayBuffer(alpha=0.7, beta=1.1, priority_key="td_error")
# rb.extend(TensorDict({"a": torch.randn(2, 3)}, batch_size=[2]))
# tensordict_sample = rb.sample(2).contiguous()
# print(tensordict_sample)
# import gymnasium as gym
# from torchrl.envs.libs.gym import GymEnv, GymWrapper

# gym_env = gym.make("Pendulum-v1")
# env = GymWrapper(gym_env)
# env = GymEnv("Pendulum-v1")


# tensordict = env.reset()

# print(
# env.rand_step(tensordict)
# )
# env = GymEnv("Pendulum-v1", frame_skip=3, from_pixels=True, pixels_only=False)
# env.reset()

# print(
# env.rand_step(tensordict)
# )
from torch import nn
from tensordict.nn import TensorDictModule

tensordict = TensorDict({"key 1": torch.randn(10, 3)}, batch_size=[10])
module = nn.Linear(3, 4)
td_module = TensorDictModule(module, in_keys=["key 1"], out_keys=["key 2"])
td_module(tensordict)
# print(tensordict)

# print(tensordict["key 2"])


from torchrl.modules import ConvNet, MLP
from torchrl.modules.models.utils import SquashDims

net = MLP(num_cells=[32, 64], out_features=4, activation_class=nn.ELU)
# print(net)
# print(net(torch.randn(10, 3)).shape)


cnn = ConvNet(
    num_cells=[32, 64],
    kernel_sizes=[8, 4],
    strides=[2, 1],
    aggregator_class=SquashDims,
)
# print(cnn)
# print(cnn(torch.randn(10, 3, 32, 32)).shape)  # last tensor is squashed

from tensordict.nn import TensorDictSequential

backbone_module = nn.Linear(5, 3)
# print(backbone_module)
backbone = TensorDictModule(
    backbone_module, in_keys=["observation"], out_keys=["hidden"]
)
# print(backbone)

actor_module = nn.Linear(3, 4)
# print(actor_module)

actor = TensorDictModule(actor_module, in_keys=["hidden"], out_keys=["action"])
# print(actor)

value_module = MLP(out_features=1, num_cells=[4, 5])
value = TensorDictModule(value_module, in_keys=["hidden", "action"], out_keys=["value"])

sequence = TensorDictSequential(backbone, actor, value)
# print(sequence)
# print(sequence.in_keys, sequence.out_keys)


tensordict = TensorDict(
    {"observation": torch.randn(3, 5)},
    [3],
)
# backbone(tensordict)
# print(backbone(tensordict)["observation"])
# print(backbone(tensordict)["hidden"])
# print(actor(tensordict)["hidden"])
# print(actor(tensordict)["action"])
# actor(tensordict)
# value(tensordict)
# print(value(tensordict)["value"])


tensordict = TensorDict(
    {"observation": torch.randn(3, 5)},
    [3],
)
sequence(tensordict)
# print(tensordict)


from torchrl.modules import Actor

base_module = nn.Linear(5, 1)
actor = Actor(base_module, in_keys=["obs"])
tensordict = TensorDict({"obs": torch.randn(5)}, batch_size=[])
# print(actor(tensordict)["action"])  # action is the default value



from tensordict.nn import (
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
)

# Probabilistic modules
from torchrl.modules import NormalParamWrapper, TanhNormal

td = TensorDict({"input": torch.randn(3, 5)}, [3])
# print(td["input"])
net = NormalParamWrapper(nn.Linear(5, 4))  # splits the output in loc and scale
module = TensorDictModule(net, in_keys=["input"], out_keys=["loc", "scale"])
td_module = ProbabilisticTensorDictSequential(
    module,
    ProbabilisticTensorDictModule(
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        return_log_prob=False,
    ),
)


# returning the log-probability
td = TensorDict({"input": torch.randn(3, 5)}, [3])
td_module = ProbabilisticTensorDictSequential(
    module,
    ProbabilisticTensorDictModule(
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        return_log_prob=True,
    ),
)
td_module(td)
# print(td)
# print(td_module(td)["action"])
# print(td["action"])


# Sampling vs mode / mean
# from torchrl.envs.utils import ExplorationType, set_exploration_type

# td = TensorDict({"input": torch.randn(3, 5)}, [3])
# print(td_module(td)["action"])

# torch.manual_seed(0)
# with set_exploration_type(ExplorationType.RANDOM):
#     td_module(td)
#     print("random:", td["action"])

# with set_exploration_type(ExplorationType.MODE):
#     td_module(td)
#     print("mode:", td["action"])

# with set_exploration_type(ExplorationType.MEAN):
#     td_module(td)
#     print("mean:", td["action"])


from torchrl.envs.libs.gym import GymEnv, GymWrapper
from torchrl.envs.utils import step_mdp
from torchrl.data import BoundedTensorSpec
from torchrl.modules import SafeModule
env = GymEnv("Pendulum-v1")

action_spec = env.action_spec
# print(action_spec)
actor_module = nn.Linear(3, 1)
actor = SafeModule(
    actor_module, spec=action_spec, in_keys=["observation"], out_keys=["action"]
)

torch.manual_seed(0)
env.set_seed(0)

max_steps = 100
tensordict = env.reset()
tensordicts = TensorDict({}, [max_steps])
for i in range(max_steps):
    actor(tensordict)
    # print(tensordict["action"])
    tensordicts[i] = env.step(tensordict)
    if tensordict["done"].any():
        break
    tensordict = step_mdp(tensordict)  # roughly equivalent to obs = next_obs

tensordicts_prealloc = tensordicts.clone()
# print("total steps:", i)
# print(tensordicts)


# equivalent
torch.manual_seed(0)
env.set_seed(0)

max_steps = 100
tensordict = env.reset()
tensordicts = []
for _ in range(max_steps):
    actor(tensordict)
    tensordicts.append(env.step(tensordict))
    if tensordict["done"].any():
        break
    # print(tensordict['observation'])
    tensordict = step_mdp(tensordict)  # roughly equivalent to obs = next_obs
    # print(tensordict['observation'])

tensordicts_stack = torch.stack(tensordicts, 0)
# print("total steps:", i)
# print(tensordicts_stack)

from torchrl.collectors import MultiaSyncDataCollector, MultiSyncDataCollector

from torchrl.envs import EnvCreator, ParallelEnv
from torchrl.envs.libs.gym import GymEnv

parallel_env = ParallelEnv(3, EnvCreator(lambda: GymEnv("Pendulum-v1")))
print(parallel_env)
create_env_fn = [parallel_env, parallel_env]
print(create_env_fn)
actor_module = nn.Linear(3, 1)
actor = TensorDictModule(actor_module, in_keys=["observation"], out_keys=["action"])


devices = ["cpu", "cpu"]

collector = MultiSyncDataCollector(
    create_env_fn=create_env_fn,  # either a list of functions or a ParallelEnv
    policy=actor,
    total_frames=240,
    max_frames_per_traj=-1,  # envs are terminating, we don't need to stop them early
    frames_per_batch=60,  # we want 60 frames at a time (we have 3 envs per sub-collector)
    storing_devices=devices,  # len must match len of env created
    devices=devices,
)

for i, d in enumerate(collector):
    if i == 0:
        x = 0
        # print(d)  # trajectories are split automatically in [6 workers x 10 steps]
    # collector.update_policy_weights_()  # make sure that our policies have the latest weights if working on multiple devices
# print(i)
# collector.shutdown()
del collector