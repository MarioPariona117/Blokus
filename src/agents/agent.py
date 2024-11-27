import gymnasium as gym
import random
import numpy as np
import pickle

def encode_board(board: np.ndarray) -> str:
    return ''.join(map(str, board.flatten()))

def decode_board(encoded_board : str, n: int) -> np.ndarray:
    board = []
    board = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            board[i, j] = int(encoded_board[i * n + j])
    return board

def create_q_table(q_table, state: str, actions: np.ndarray):
    if state not in q_table:
        q_table[state] = {action: 0 for action in actions}
    

def get_q_value(q_table, state: str, action: int) -> float:
    return q_table[state][action]

def set_q_value(q_table, state: str, action: int, value: float):
    q_table[state][action] = value

def argmax(q_table, state: str) -> int:
    return max(q_table[state], key=q_table[state].get)

def maxi(q_table, state: str) -> float:
    if len(q_table[state].values()) == 0:
        return 0
    return max(q_table[state].values())

def random_policy(actions):
    if len(actions) == 0:
        return None
    return random.choice(actions)

def maximize_policy(action_to_value, env, actions):
    if not actions:
        return None
    return max(actions, key=lambda action: action_to_value(action))

def from_q_agent_policy(env, actions):
    return argmax(env.q_table, env.obs["state"])

class Agent:
    policy_map = {
        "random": random_policy,
        "maximize": maximize_policy,
        "q_agent": from_q_agent_policy
    }
    def __init__(self, name="Agent", policy=None, dictionary=None):
        self.name = name
        self.policy = self.policy_map[policy]  # Function to determine the agent's actions
        if dictionary is not None:
            self.dictionary = dictionary

    def get_action(self, env: gym.Env, actions):
        action = self.policy(env, actions) 
        return action
    

class QL_Agent(Agent):
    def __init__(self, name="Agent", dictionary=None):
        self.name = name
        if dictionary is not None:
            with open(dictionary, 'rb') as file:
                self.q_table = pickle.load(file)
            print("Q-table loaded")

    def get_action(self, obs):
        return argmax(self.q_table, encode_board(obs["state"]))