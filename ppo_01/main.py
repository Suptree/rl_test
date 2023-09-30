# File: main.py

import torch
import matplotlib.pyplot as plt

from environment_manager import EnvironmentManager
from network_manager import NetworkManager
from trainer import Trainer

if __name__ == "__main__":
    # Common parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_cells = 256  # This is just an example; adjust as needed.
    frame_skip = 1  # Number of frames to skip (adjust based on your requirement)
    frames_per_batch = 1000 // frame_skip # 各バッチごとのフレーム数を示します。これは、エージェントが一度に収集するデータの量を示し

    # Create environment
    env_manager = EnvironmentManager(device, frame_skip)
    env = env_manager.setup_environment()

    # Setup networks
    network_manager = NetworkManager(num_cells, env, device)
    policy_module, value_module = network_manager.setup_networks()
    print("Running policy:", policy_module(env.reset()))
    print("Running value:", value_module(env.reset()))
    # policy_module(torch.tensor(dummy_observation).unsqueeze(0).to(device))  # ダミーオブザベーションでpolicy_moduleを初期化
    # Training configuration
    trainer_config = {
        "frames_per_batch": 1000 // frame_skip,
        "total_frames":  50_000 // frame_skip,  # Total frames to train on. Adjust as needed.
        "num_epochs": 10,  # Number of epochs per batch. Adjust as needed.
        "sub_batch_size": 64,  # Sub batch size for optimization. Adjust as needed.
        "clip_epsilon": 0.2,  # PPO's epsilon for clipping. Adjust as needed.
        "gamma": 0.99,  # Discount factor. Adjust as needed.
        "lmbda": 0.95,  # Lambda for GAE. Adjust as needed.
        "entropy_eps": 1e-4,  # Entropy bonus. Adjust as needed.
        "max_grad_norm": 1.0,  # Max gradient norm. Adjust as needed.
        "lr": 3e-4,  # Learning rate. Adjust as needed.
    }

    # Train

    trainer = Trainer(env, policy_module, value_module, device, trainer_config)
    logs = trainer.train()

    # Post-training: results visualization
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
    plt.plot(logs["reward"], label="Reward")
    plt.plot(logs.get("eval reward", []), label="Eval Reward", linestyle="--")
    plt.legend()
    plt.title("Reward over training")

    plt.subplot(2, 2, 4)
    plt.plot(logs["step_count"], label="Step Count")
    plt.plot(logs.get("eval step_count", []), label="Eval Step Count", linestyle="--")
    plt.legend()
    plt.title("Step count over training")

    plt.tight_layout()
    plt.show()
