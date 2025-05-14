import torch
import torch.nn as nn
import torch.optim as optim

from itertools import accumulate, chain
from collections import deque
from typing import List
from tqdm import tqdm
import random
import wandb
import copy
import json
import os

from gymnasium_env import BlokusAction, BlokusEnv, SingleAgentBlokusEnv

from src.agents import Agent, RandomAgent
from src.agents.qnetwork.qnetwork_agent import QNetworkAgent

from gymnasium import Wrapper

class TrainDQN:
    def __init__(
        self,
        device: str,
        agent: QNetworkAgent,
        player_turn: int,
        wrappers: List[Wrapper],
        buffer_size: int,
        batch_size: int,
        gamma: float,
        lr: float,
        target_update_freq: int,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.99,
        min_epsilon: float = 0.1,
        use_wandb: bool = False,
        wandb_project: str = "dqn-training",
    ):
        """
        Trains the DQN agent against an embedded agent using the DQN algorithm.

        Args:
            device (str): Device to run the training on ('cpu', 'mps', or 'cuda').
            agent (QNetworkAgent): The DQN agent to be trained.
            player_turn (int): The player turn for the agent (0, 1, 2, or 3).
            wrappers (List[Wrapper]): List of wrappers to apply to the environment. They will be applied in order.
            buffer_size (int): Max size of the replay buffer.
            batch_size (int): Number of samples per batch.
            gamma (float): Discount factor for future rewards.
            lr (float): Learning rate for the optimizer.
            target_update_freq (int): Frequency of syncing target network with policy network.
            epsilon (float, optional): Initial epsilon value for exploration in the epsilon-greedy policy. Defaults to 1.0.
            epsilon_decay (float): Decay rate for epsilon.
            min_epsilon (float, optional): Minimum epsilon value for exploration. Defaults to 0.1.
            use_wandb (bool, optional): Whether to use Weights & Biases for logging. Defaults to False.
            wandb_project (str, optional): Name of the Weights & Biases project. Defaults to "dqn-training".
        """
        # Core parameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.agent = agent
        self.board_size = agent.board_size
        self.player_turn = player_turn

        # Models
        self.target_net = copy.deepcopy(self.agent.policy_net)  # The target network that gets updated
        self.target_net.eval()  # Target network is not trained directly

        env = BlokusEnv(
            board_size=self.board_size
        )
        for wrapper in wrappers:
            env = wrapper(env)

        self.env = SingleAgentBlokusEnv(
            base_env=env,
            hidden_agents=[None] * 3,
            player_turn=self.player_turn,
        )

        # Optimizer
        self.optimizer = optim.Adam(self.agent.policy_net.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)

        # Step tracking for target updates
        self.target_update_freq = target_update_freq
        self.step_count = 0

        # Wandb setup
        self.wandb_enabled = use_wandb
        if use_wandb:
            self.run = wandb.init(
                project=wandb_project,
                config={
                    "board_size": self.agent.board_size,
                    "gamma": gamma,
                    "lr": lr,
                    "batch_size": batch_size,
                    "buffer_size": buffer_size,
                    "target_update_freq": target_update_freq,
                    "epsilon_decay": epsilon_decay,
                    "min_epsilon": min_epsilon,
                    "player_turn": player_turn,
                },
            )

    def collect_trajectories(self, max_steps: int, pbar: bool, hidden_agent: Agent = RandomAgent()):
        self.agent.eval()
        self.env.hidden_agents[3 - self.player_turn] = hidden_agent
        obs, info = self.env.reset()

        prange = range(max_steps)
        if pbar: 
            prange = tqdm(prange, desc="Collecting trajectories")

        for _ in prange:
            if random.random() < self.epsilon:
                action_id = random.choice(obs["possible_actions"])
            else:
                q_values = self.agent.get_q_values(obs)
                action_idx = q_values.argmax().item()
                action_id = obs["possible_actions"][action_idx]

            next_obs, reward, term, trunc, info = self.env.step(action_id)

            assert not trunc
            done = term

            self.replay_buffer.append((obs, action_id, reward, next_obs, done))

            if done:
                obs, info = self.env.reset()
            else: 
                obs = next_obs
                
        self.agent.train()

    def optimize_model(self):
        self.agent.train()
        """Performs a single optimization step on the policy network."""
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        obss, actions, rewards, next_obss, dones = zip(*batch)

        encoded_actions = self.target_net.cat([
            self.target_net.encode(obs, [action], device=self.device)
            for obs, action in zip(obss, actions)
        ])
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        current_q_values = self.agent.policy_net(encoded_actions)

        with torch.no_grad():
            repeated_encoded_actions = self.target_net.cat([
                self.target_net.encode(next_obs, next_obs["possible_actions"], device=self.device)
                for next_obs in next_obss
            ])
            aux = self.target_net(repeated_encoded_actions)

            accumulated_idx = list(accumulate(chain([0], [len(obs["possible_actions"]) for obs in next_obss])))

            max_next_q_values = torch.tensor([
                torch.max(aux[accumulated_idx[idx]:accumulated_idx[idx + 1]]) 
                if accumulated_idx[idx] != accumulated_idx[idx + 1] else 0.0 # if empty, set to 0
                for idx in range(self.batch_size)
            ]).to(self.device)
            next_q_values = max_next_q_values * (1 - dones)
            target_q_values = rewards + self.gamma * next_q_values 

        loss = nn.MSELoss()(current_q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        logs = {
            "loss": loss.item(),
            "epsilon": self.epsilon,
        }

        # Update target network
        self.step_count += 1

        # Log loss to Wandb
        if self.wandb_enabled:
            self.run.log(logs, step=self.step_count)

        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.agent.policy_net.state_dict())

        self.agent.eval()
        return logs

    def save_model(self):
        """Saves the model to the given path."""
        self.agent.save_model()
        self.save_training_config()
        torch.save(self.agent.policy_net.state_dict(), self.model_path)

    def save_training_config(self):
        """Saves the configuration to a JSON file."""
        training_config = {
            "board_size": self.board_size,
            "player_turn": self.player_turn,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "min_epsilon": self.min_epsilon,
            "batch_size": self.batch_size,
            "buffer_size": len(self.replay_buffer),
            "target_update_freq": self.target_update_freq,
            "lr": self.optimizer.param_groups[0]["lr"],
            "device": str(self.device),
            "model_class": self.agent.model_class.__name__,
        }
        config_path = os.path.join(self.agent.dir, "train_config.json")
        with open(config_path, "w") as f:
            json.dump(training_config, f, indent=2)

    def update_epsilon(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
