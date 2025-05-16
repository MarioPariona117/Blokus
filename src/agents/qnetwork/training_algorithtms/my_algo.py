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

from .base_trainer import BaseTrainer

class MyAlgo(BaseTrainer):
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
        *args, 
        **kwargs
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
        super().__init__(agent=agent, *args, **kwargs)
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
        self.opponent_agent = copy.deepcopy(self.agent)
        
        self.load_opponent_model()
        self.opponent_target_net.eval()

        # Blokus environment
        env = BlokusEnv(
            board_size=self.board_size
        )
        for wrapper in wrappers:
            env = wrapper(env)

        self.env = env
        self.obs, info = self.env.reset()

        # Optimizer
        self.optimizer = optim.Adam(
            chain(self.agent.policy_net.parameters(), self.opponent_agent.policy_net.parameters()), lr=lr
        )

        initial_lr = lr
        min_lr = lr / 10
        factor = self.epsilon_decay
        def lr_lambda(epoch):
            if factor ** epoch < min_lr / initial_lr:
                return 1
            elif factor ** (epoch+1) < min_lr / initial_lr:
                return min_lr / factor ** epoch # no, the part when epoch is too big missing
            else:
                return factor

        self.lr_scheduler = optim.lr_scheduler.MultiplicativeLR(
            self.optimizer, lr_lambda=lr_lambda
        )

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
            self.opponent_agent.policy_net.load_state_dict(torch.load(opponent_model_path, map_location=self.device))
        else:
            # self.opponent_policy_net = copy.deepcopy(self.agent.policy_net)
            print(f"Opponent model not found at {opponent_model_path}")
        self.opponent_target_net = copy.deepcopy(self.opponent_agent.policy_net)
        
    def collect_trajectories(self, max_steps: int, pbar: bool):
        self.agent.eval()
        # self.obs, info = self.env.reset()

        prange = range(max_steps)
        if pbar: 
            prange = tqdm(prange, desc="Collecting trajectories")

        for _ in prange:
            random_num = self.rng.random()

            # Conditional for when to be random for my agent (epsilon greedy) and opponent (stochasticity field)
            if (self.obs["current_player"] != self.player_turn and random_num < self.opponent_stochasticity) or \
            (self.obs["current_player"] == self.player_turn and random_num < self.epsilon): 
                action_id = self.rng.choice(self.obs["possible_actions"])
            else:
                with torch.no_grad(): 
                    agent = self.agent if self.obs["current_player"] == self.player_turn else self.opponent_agent
                    q_values = agent.get_q_values(self.obs).detach()
                    
                    if self.obs["current_player"] == 1:
                        action_idx = q_values.argmax().item()
                    else: 
                        action_probs = F.softmax(q_values * 10, dim=0)
                        action_idx = torch.multinomial(action_probs, 1).item()
                action_id = self.obs["possible_actions"][action_idx]

            next_obs, reward, term, trunc, info = self.env.step(action_id)

            assert not trunc
            done = term

            self.replay_buffer.append((self.obs, action_id, reward, next_obs, done))

            self.obs = next_obs

            if done:
                self.obs, info = self.env.reset()
                
    def optimize_model(self):
        """Performs a single optimization step on the policy network."""
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.rng.sample(self.replay_buffer, self.batch_size)
        obss, actions, rewards, next_obss, dones = zip(*batch)

        # Convert to tensors
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        my_turns = torch.tensor([obs["current_player"] == self.player_turn for obs in obss], dtype=torch.bool, device=self.device)
        my_next_turns = torch.tensor([obs["current_player"] == self.player_turn for obs in next_obss], dtype=torch.bool, device=self.device)
        switches = torch.tensor(
            [obs["current_player"] != next_obs["current_player"] for obs, next_obs in zip(obss, next_obss)],
            dtype=torch.float32
        , device=self.device)

        # Compute current Q values
        encoded_actions = self.target_net.cat([
            self.target_net.encode(obs, [action], device=self.device)
            for obs, action in zip(obss, actions)
        ])
        my_current_q_values = self.agent.policy_net(encoded_actions)
        opponent_current_q_values = self.opponent_agent.policy_net(encoded_actions)
        current_q_values = torch.where(my_turns, my_current_q_values, opponent_current_q_values)

        with torch.no_grad():
            repeated_encoded_actions = self.target_net.cat([
                self.target_net.encode(next_obs, next_obs["possible_actions"], device=self.device)
                for next_obs in next_obss
            ])

            my_aux = self.target_net(repeated_encoded_actions)
            opponent_aux = self.opponent_target_net(repeated_encoded_actions)

            repeated_my_next_turns = torch.cat([my_next_turns[idx].expand(len(next_obs["possible_actions"])) for idx, next_obs in enumerate(next_obss)])
            aux = torch.where(repeated_my_next_turns, my_aux, opponent_aux)

            accumulated_idx = list(accumulate(chain([0], [len(obs["possible_actions"]) for obs in next_obss])))

            max_next_q_values = torch.tensor([
                torch.max(aux[accumulated_idx[idx]:accumulated_idx[idx + 1]]) 
                if accumulated_idx[idx] != accumulated_idx[idx + 1] else 0.0 # if empty, set to 0
                for idx in range(self.batch_size)
            ], device=self.device)
            next_q_values = max_next_q_values * (1 - dones) * (1 - 2 * switches)
            target_q_values = rewards + self.gamma * next_q_values 

        # Compute loss
        # loss = F.mse_loss(current_q_values, target_q_values)
        loss = F.smooth_l1_loss(current_q_values, target_q_values, beta=0.5)
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
        self.opponent_agent.save_model(path=os.path.join(self.agent.dir, "opponent_model.pth"))
        self.save_traning_config()

    def save_traning_config(self):
        """Saves the training configuration to a JSON file."""
        training_config = {
            # Environment & agent setup
            "board_size": self.board_size,
            "player_turn": self.player_turn,

            # Training hyperparameters
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "min_epsilon": self.min_epsilon,
            "batch_size": self.batch_size,
            "buffer_size": len(self.replay_buffer),
            "lr": self.optimizer.param_groups[0]["lr"],
            "target_update_freq": self.target_update_freq,
            "opponent_stochasticity": self.opponent_stochasticity,

            # Meta info
            "device": str(self.device),
            "model_class": self.agent.model_class.__name__,
            "iterations": self.step_count,

            # Logging / tracking
            "project-name": self.run.project,
            "run-id": self.run.id,
        }
        config_path = os.path.join(self.agent.dir, "training_config.json")
        with open(config_path, "w") as f:
            json.dump(training_config, f, indent=2)

    def update_epsilon(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
