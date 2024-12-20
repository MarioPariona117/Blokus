import random
import pickle
import numpy as np
from .agent import Agent
from src.utils import encode_board_string, decode_board_string
class QL_Agent(Agent):
    def __init__(self, name="QL_Agent", q_table_path=None, alpha=0.01, gamma=0.995, epsilon=1.0, epsilon_decay=1, min_epsilon=0.01):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.name = name
        self.q_table_path = q_table_path
        self.load_q_table()

    def get_action(self, env, obs):
        state = encode_board_string(obs["state"])
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(obs["possible_actions"])
        else:
            action = self.argmax(state)
        return action
    
    def save_q_table(self):
        with open(self.q_table_path, 'wb') as file:
            pickle.dump(self.q_table, file)
        
    def load_q_table(self):
        with open(self.q_table_path, 'rb') as file:
            self.q_table = pickle.load(file)

    def create_q_table(self, state: str, actions: np.ndarray):
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in actions}

    def get_q_value(self, state: str, action: int) -> float:
        return self.q_table[state][action]

    def set_q_value(self, state: str, action: int, value: float):
        self.q_table[state][action] = value
        
    def argmax(self, state: str) -> float:
        if state not in self.q_table:
            return None
        return max(self.q_table[state], key=self.q_table[state].get)