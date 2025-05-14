import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from itertools import accumulate, chain
from collections import deque
from typing import List
from tqdm import tqdm
import random
import wandb
import copy
import os
import json

from gymnasium_env import BlokusAction, BlokusEnv, SingleAgentBlokusEnv
from gymnasium import Wrapper

from src.agents.qnetwork.qnetwork_agent import QNetworkAgent
from src.agents import Agent, RandomAgent

class MyAlgo:
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
        opponent_stochasticity: float = 0.25,
        use_wandb: bool = False,
        wandb_project: str = "myalgo-training",
    ):
        """
        Initializes the MyAlgo class.
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
            opponent_stochasticity (float, optional): Probability of the opponent taking a random action. Defaults to 0.25.
            use_wandb (bool, optional): Whether to use Weights & Biases for logging. Defaults to False.
            wandb_project (str, optional): Name of the Weights & Biases project. Defaults to "myalgo-training".
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
        self.opponent_stochasticity = opponent_stochasticity

        # Agent setup
        self.target_net = copy.deepcopy(self.agent.policy_net)
        self.target_net.eval()

        # Opponent Network setup
        self.opponent_policy_net = copy.deepcopy(self.agent.policy_net)
        self.load_opponent_model()
        self.opponent_target_net.eval()

        # Blokus environment
        env = BlokusEnv(
            board_size=self.board_size
        )
        for wrapper in wrappers:
            env = wrapper(env)

        self.env = env

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

    def load_opponent_model(self) -> None:
        """Loads the opponent model from the specified path."""
        opponent_model_path = os.path.join(self.agent.dir, "opponent_model.pth")
        if os.path.exists(opponent_model_path):
            self.opponent_policy_net.load_state_dict(torch.load(opponent_model_path, map_location=self.device))
        else:
            self.opponent_policy_net = copy.deepcopy(self.agent.policy_net)
            print(f"Opponent model not found at {opponent_model_path}")
        self.opponent_target_net = copy.deepcopy(self.opponent_policy_net)
        
    def collect_trajectories(self, max_steps: int, pbar: bool):
        self.agent.eval()
        obs, info = self.env.reset()

        prange = range(max_steps)
        if pbar: 
            prange = tqdm(prange, desc="Collecting trajectories")

        for _ in prange:
            random_num = random.random()

            # Conditional for when to be random for my agent (epsilon greedy) and opponent (stochasticity field)
            if (obs["current_player"] != self.player_turn and random_num < self.opponent_stochasticity) or \
            (obs["current_player"] == self.player_turn and random_num < self.epsilon): 
                action_id = random.choice(obs["possible_actions"])
            else:
                with torch.no_grad(): 
                    q_values = self.agent.get_q_values(obs).detach()
                    
                    if obs["current_player"] == 1:
                        action_idx = q_values.argmax().item()
                    else: 
                        action_probs = F.softmax(q_values, dim=0)
                        action_idx = torch.multinomial(action_probs, 1).item()
                action_id = obs["possible_actions"][action_idx]

            next_obs, reward, term, trunc, info = self.env.step(action_id)

            assert not trunc
            done = term

            self.replay_buffer.append((obs, action_id, reward, next_obs, done))

            obs = next_obs

            if done:
                obs, info = self.env.reset()
                
    def optimize_model(self):
        """Performs a single optimization step on the policy network."""
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        obss, actions, rewards, next_obss, dones = zip(*batch)

        # Convert to tensors
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        my_turns = torch.tensor([obs["current_player"] == self.player_turn for obs in obss], dtype=torch.bool).to(self.device)
        my_next_turns = torch.tensor([obs["current_player"] == self.player_turn for obs in next_obss], dtype=torch.bool).to(self.device)
        switches = torch.tensor(
            [obs["current_player"] != next_obs["current_player"] for obs, next_obs in zip(obss, next_obss)],
            dtype=torch.float32
        ).to(self.device)

        # Compute current Q values
        encoded_actions = self.target_net.cat([
            self.target_net.encode(obs, [action], device=self.device)
            for obs, action in zip(obss, actions)
        ])
        my_current_q_values = self.agent.policy_net(encoded_actions)
        opponent_current_q_values = self.opponent_policy_net(encoded_actions)
        current_q_values = torch.where(my_turns, my_current_q_values, opponent_current_q_values)

        with torch.no_grad():
            repeated_encoded_actions = self.target_net.cat([
                self.target_net.encode(next_obs, next_obs["possible_actions"], device=self.device)
                for next_obs in next_obss
            ])

            my_aux = self.target_net(repeated_encoded_actions)
            opponent_aux = self.opponent_target_net(repeated_encoded_actions)

            repeated_my_next_turns = torch.cat([my_next_turns[idx].expand(len(next_obs["possible_actions"])) for idx, next_obs in enumerate(next_obss)]).to(self.device)
            aux = torch.where(repeated_my_next_turns, my_aux, opponent_aux)

            accumulated_idx = list(accumulate(chain([0], [len(obs["possible_actions"]) for obs in next_obss])))

            max_next_q_values = torch.tensor([
                torch.max(aux[accumulated_idx[idx]:accumulated_idx[idx + 1]]) 
                if accumulated_idx[idx] != accumulated_idx[idx + 1] else 0.0 # if empty, set to 0
                for idx in range(self.batch_size)
            ]).to(self.device)
            next_q_values = max_next_q_values * (1 - dones) * (1 - 2 * switches)
            target_q_values = rewards + self.gamma * next_q_values 

        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        logs = {
            "loss": loss.item(),
            "epsilon": self.epsilon,
        }

        self.step_count += 1
        
        # Log loss to Wandb
        if self.wandb_enabled:
            self.run.log(logs, step=self.step_count)

        # Update target network
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.agent.policy_net.state_dict())

        self.agent.eval()
        return logs

    def save_model(self):
        """Saves the model to the given path."""
        self.agent.save_model()
        self.save_opponent_model()
        self.save_traning_config()

    def save_traning_config(self):
        """Saves the training configuration to a JSON file."""
        config = {
            "board_size": self.board_size,
            "gamma": self.gamma,
            "lr": self.optimizer.param_groups[0]["lr"],
            "batch_size": self.batch_size,
            "buffer_size": len(self.replay_buffer),
            "target_update_freq": self.target_update_freq,
            "epsilon_decay": self.epsilon_decay,
            "min_epsilon": self.min_epsilon,
            "player_turn": self.player_turn,
        }
        config_path = os.path.join(self.agent.dir, "training_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    def save_opponent_model(self):
        """Saves the opponent model to the specified path."""
        opponent_model_path = os.path.join(self.agent.dir, "opponent_model.pth")
        torch.save(self.opponent_policy_net.state_dict(), opponent_model_path)
        torch.save(self.opponent_target_net.state_dict(), opponent_model_path)

    def update_epsilon(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
