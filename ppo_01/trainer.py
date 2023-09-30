# File: trainer.py

from collections import defaultdict
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.envs.utils import set_exploration_mode


class Trainer:
    def __init__(self, env, policy_module, value_module, device, config):
        self.env = env
        self.policy_module = policy_module
        self.value_module = value_module
        self.device = device
        self.config = config

    def train(self):
        # Extract configuration
        frames_per_batch = self.config["frames_per_batch"]
        total_frames = self.config["total_frames"]
        num_epochs = self.config["num_epochs"]
        sub_batch_size = self.config["sub_batch_size"]
        clip_epsilon = self.config["clip_epsilon"]
        gamma = self.config["gamma"]
        lmbda = self.config["lmbda"]
        entropy_eps = self.config["entropy_eps"]
        max_grad_norm = self.config["max_grad_norm"]
        lr = self.config["lr"]

        collector = SyncDataCollector(
            self.env,
            self.policy_module,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            split_trajs=False,
            device=self.device,
        )

        replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(frames_per_batch),
            sampler=SamplerWithoutReplacement(),
        )

        advantage_module = GAE(
            gamma=gamma, lmbda=lmbda, value_network=self.value_module, average_gae=True
        )

        loss_module = ClipPPOLoss(
            actor=self.policy_module,
            critic=self.value_module,
            advantage_key="advantage",
            clip_epsilon=clip_epsilon,
            entropy_bonus=bool(entropy_eps),
            entropy_coef=entropy_eps,
            critic_coef=1.0,
            gamma=gamma,
            loss_critic_type="smooth_l1",
        )

        optim = Adam(loss_module.parameters(), lr)
        scheduler = CosineAnnealingLR(optim, total_frames // frames_per_batch, 0.0)

        logs = defaultdict(list)
        pbar = tqdm(total=total_frames)

        for i, tensordict_data in enumerate(collector):
            for _ in range(num_epochs):
                advantage_module(tensordict_data)
                data_view = tensordict_data.reshape(-1)
                replay_buffer.extend(data_view.cpu())

                for _ in range(frames_per_batch // sub_batch_size):
                    subdata = replay_buffer.sample(sub_batch_size)
                    loss_vals = loss_module(subdata.to(self.device))

                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                    loss_value.backward()
                    clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                    optim.step()
                    optim.zero_grad()

            logs["reward"].append(tensordict_data["next", "reward"].mean().item())
            logs["step_count"].append(tensordict_data["step_count"].max().item())
            logs["lr"].append(optim.param_groups[0]["lr"])

            pbar.set_description(
                f"reward: {logs['reward'][-1]:.4f}, step count: {logs['step_count'][-1]}, lr: {logs['lr'][-1]:.4f}"
            )

            if i % 10 == 0:
                with set_exploration_mode("mean"), torch.no_grad():
                    eval_rollout = self.env.rollout(1000, self.policy_module)
                    logs["eval reward"].append(
                        eval_rollout["next", "reward"].mean().item()
                    )
                    logs["eval reward (sum)"].append(
                        eval_rollout["next", "reward"].sum().item()
                    )
                    logs["eval step_count"].append(
                        eval_rollout["step_count"].max().item()
                    )
                    del eval_rollout

            scheduler.step()

        return logs
