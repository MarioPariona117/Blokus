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
    if state not in q_table:
        return None
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
    def __init__(self, name="Agent", policy="random", simple_policy=None):
        self.name = name
        self.policy = self.policy_map[policy]  # Function to determine the agent's actions
        self.simple_policy = simple_policy  # Function to determine the agent's actions if the policy is "random"
    
    def get_action(self, env, obs):
        actions = obs["possible_actions"]
        next_obss = [env.get_next_obs(env, action) for action in actions]
        return self.choose_action(obs, actions, next_obss)
    
    def choose_action(self, obs, actions, next_obss):
        action = self.policy(obs, actions, next_obss)
        return action

class RandomAgent(Agent):
    def __init__(self, name="Agent"):
        super().__init__(name, "random", random_policy)

    def get_action(self, env, obs):
        actions = obs["possible_actions"]
        return random.choice(actions)
    
class FunctionAgent(Agent):
    def __init__(self, name="Agent", func=None):
        self.name = name
        self.func = func # (obs, action, next_obs) -> float

    def choose_action(self, obs, actions, next_obss):
        action_values = np.array(np.map(lambda x: self.func(obs, x[0], x[1]), zip(actions, next_obss)))
        return actions[np.argmax(action_values)]
    
class SimpleFunctionAgent(Agent):
    def __init__(self, name="Agent", func=None):
        self.name = name
        self.func = func # (actions) -> action

    def get_action(self, env, obs):
        actions = obs["possible_actions"]
        return self.func(env, actions)
    
class QL_Agent(Agent):
    def __init__(self, name="Agent", dictionary=None):
        self.name = name
        if dictionary is not None:
            with open(dictionary, 'rb') as file:
                self.q_table = pickle.load(file)
            print("Q-table loaded")

    def get_action(self, env, obs):
        return argmax(self.q_table, encode_board(obs["state"]))
    
class MiniMaxAgent(Agent):
    def __init__(self, name="Agent", depth=1, board_size=7, cache_dir = "/Users/mario/Documents/proj/cam/Blokus/src/agents/cache"):
        self.name = name
        self.depth = depth
        self.board_size = board_size
        self.env = gym.make('gymnasium_env/Blokus-v0', board_size=board_size, num_players=2, render_mode='console', render_scale=10, disable_env_checker=True, neighborhood_dir="/Users/mario/Documents/proj/cam/Blokus/gymnasium_env/envs/auxiliary/pre_neighbors", mode = "good")
        self.env = self.env.unwrapped
        self.env.order_enforce = False
        self.cache_file = f"{cache_dir}/minimax_depth{self.depth}_bz{board_size}.pkl"
        self.load_cache()
        # print(self.cache)

    def get_action(self, env, obs):
        encoded_board = encode_board(obs["state"])
        if encoded_board in self.cache:
            return random.choice(self.cache[encoded_board])
        state = env.capture_state()
        self.env.restore_state(state)
        good_actions = self.minimax(obs, self.depth)[0]
        self.cache[encoded_board] = good_actions
        return random.choice(good_actions)
    
    def save_cache(self):
        if self.cache_file is not None:
            with open(self.cache_file, 'wb') as file:
                pickle.dump(self.cache, file)

    def load_cache(self):
        if self.cache_file is not None:
            with open(self.cache_file, 'rb') as file:
                self.cache = pickle.load(file)
        print(f"Cache loaded for MiniMaxAgent with depth {self.depth} and board size {self.board_size}")

    def minimax(self, obs, depth):
        if depth == 0:
            return None, 0
        actions = obs["possible_actions"]
        value = np.zeros(len(actions))
        best_value = -np.inf
        if depth == 1:
            for idx, action in enumerate(actions):
                _, _, pid, pt = self.env._action_to_tuple(action)
                psz = len(self.env.pieces[pid].transformations[pt].shape)
                value[idx] = psz
                if psz > best_value:
                    best_value = psz
        else:
            state = self.env.capture_state()
            for idx, action in enumerate(actions):
                _, _, pid, pt = self.env._action_to_tuple(action)
                psz = len(self.env.pieces[pid].transformations[pt].shape)
                new_obs, new_reward, term, trunc, _ = self.env.step(action)
                assert new_reward == psz
                assert not trunc
                if term:
                    value1 = psz
                else:
                    value1 = -self.minimax(new_obs, depth - 1)[1] + psz
                self.env.restore_state(state)
                value[idx] = value1
                if value1 > best_value:
                    best_value = value1
        if depth == self.depth:
            good_actions = actions[value == best_value]
        else:
            good_actions = None
        return good_actions, best_value