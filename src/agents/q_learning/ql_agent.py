import random
import pickle
import numpy as np
import os
from datetime import datetime

from src.agents import Agent
from src.utils import encode_board_string, decode_board_string
from proj_config import MODEL_DIR

class QL_Agent(Agent):
    def __init__(
            self, 
            board_size,
            name="QL_Agent", 
            model_folder: str = None,
            dir: str = None,
            alpha=0.01, 
            gamma=0.995, 
            epsilon=1.0, 
            epsilon_decay=0.9995,
            min_epsilon=0.01, 
            parameter_update_frequency=1000, 
            estimated_steps=18,
            training=True,
            *args, 
            **kwargs
        ):
        self.left_reward_estimate, self.right_reward_estimate = -5, 1
        self.max_alpha, self.min_alpha = 0.1, 0.01
        self.left_accuracy, self.right_accuracy = 0.5, 1.0
        self.theta = 0.95
        self.alpha = alpha  # Learning rate
        # self.beta = beta
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.name = name
        self.board_size = board_size  

        if dir is None:
            if model_folder is None:
                model_folder = self.generate_model_folder()

            self.dir = os.path.join(
                MODEL_DIR,
                self.__class__.__name__.lower(),
                f"board_{self.board_size}",
                model_folder
            )
        else:
            self.dir = dir
        self.load_model()

        os.makedirs(name=self.dir, exist_ok=True)

        self.learns = 0
        self.parameter_update_frequency = parameter_update_frequency
        self.estimated_steps = estimated_steps
        self.training = training
        super().__init__(name=name, *args, **kwargs)

    def generate_model_folder(self) -> str:
        """Generates a file name based on the current date and time."""
        now = datetime.now()
        return f"{now.strftime('%Y%m%d_%H%M%S')}"

    def estimated_reward_scale(self, estimated_reward):
        """Value output between 1 and sqrt(self.max_alpha / self.alpha)"""
        max_value = (self.max_alpha / self.alpha) ** 0.5
        estimated_reward = np.clip(estimated_reward, self.left_reward_estimate, self.right_reward_estimate)
        ret = max_value ** ((self.right_reward_estimate - estimated_reward) / (self.right_reward_estimate - self.left_reward_estimate))
        assert 1 <= ret <= max_value, f"ret: {ret}, max_value: {max_value}, right_reward_estimate: {self.right_reward_estimate}, estimated_reward: {estimated_reward}"
        return ret
    
    def step_scale(self, steps):
        """Value output between 1 and sqrt(self.max_alpha / self.alpha)"""
        max_value = (self.max_alpha / self.alpha) ** 0.5
        ret = 1 + (max_value - 1) * ((1 - self.theta ** steps) / (1 - self.theta ** self.estimated_steps))
        assert 1 <= ret <= max_value, f"ret: {ret}, max_value: {max_value}, theta: {self.theta}, steps: {steps}, estimated_steps: {self.estimated_steps}"
        return ret
    
    def get_action(self, env, obs):
        state = encode_board_string(obs["state"])
        action = self.argmax(state)
        if self.rng.uniform(0, 1) < self.epsilon or not action:
            action = self.rng.choice(obs["possible_actions"])  # Random action if exploring or unseen state
        return action
    
    def save_q_table(self, path: str | None = None) -> None:
        path = path or os.path.join(self.dir, "q_table.pkl")
        tmp_file_path = f"{path}.tmp"
    
        # Save the Q-table to the temporary file
        with open(tmp_file_path, 'wb') as tmp_file:
            pickle.dump(self.q_table, tmp_file)

        try:
            os.rename(tmp_file_path, self.q_table_path)
            print(f"Q-table successfully saved to {self.q_table_path}")
        except Exception as e:
            print(f"Error renaming file: {e}")
        
    def load_model(self, path: str| None = None) -> None:
        path = path or os.path.join(self.dir, "q_table.pkl")
        if os.path.exists(path):
            try:
                with open(path, 'rb') as file:
                    self.q_table = pickle.load(file)
                print("Hey, Q-table loaded from", path)
            except Exception as e:
                print(f"Q-table not found at {path}, initializing new Q-table.")
                raise e
        else:
            print(f"Q-table file not found at {path}, initializing new Q-table.")
            self.q_table = {}

    def create_q_table(self, state: str):
        if state not in self.q_table:
            self.q_table[state] = {}

    def get_q_value(self, state: str, action: int) -> float:
        self.create_q_table(state)
        return self.q_table[state].get(action, 0)

    def set_q_value(self, state: str, action: int, value: float):
        self.create_q_table(state)
        self.q_table[state][action] = value
        
    def argmax(self, state: str) -> float:
        if state not in self.q_table or self.q_table[state] == {}:
            return None
        return max(self.q_table[state], key=self.q_table[state].get)
    
    def learn(self, state: str, action: int, reward: float, next_state: str, done: bool, steps: int):
        if not self.training:
            return
        assert steps >= 0, f"steps: {steps}"
        next_max_q_value = self.maxi(next_state)
        estimated_reward = reward + self.gamma * (1 - done) * next_max_q_value
        estimated_reward_scale = self.estimated_reward_scale(estimated_reward)
        step_scale = self.step_scale(steps)
        # lr = self.alpha * estimated_reward_scale * step_scale
        lr = self.alpha 
        assert 0 < lr <= self.max_alpha, f"lr: {lr}, alpha: {self.alpha}, estimated_reward_scale: {estimated_reward_scale}, step_scale: {step_scale}"
        current_q_value = self.get_q_value(state, action)
        updated_q_value = current_q_value + lr * (estimated_reward - current_q_value)
        self.set_q_value(state=state, action=action, value=updated_q_value)

        self.learns += 1
        if self.learns % self.parameter_update_frequency == 0:
            self.update_parameters()

    def update_parameters(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def maxi(self, state: str) -> float:
        return self.get_q_value(state, self.argmax(state))

    def eval(self):
        assert self.training, "Agent must be in training mode to switch to test mode."
        self.epsilon_backup = self.epsilon
        self.epsilon = 0
        self.training = False

    def train(self):
        assert not self.training, "Agent must be in test mode to switch to training mode."
        self.training = True
        self.epsilon = self.epsilon_backup
