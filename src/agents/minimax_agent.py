import gymnasium as gym
import gymnasium_env
from gymnasium_env.envs import BlokusAction
import random
import pickle
import numpy as np
import os
from .agent import Agent
from src.utils import encode_board_string, decode_board_string

class MiniMaxAgent(Agent):
    def __init__(self, board_size, name="Agent", depth=100, use_cache=True, cache_dir = "/Users/mario/Documents/proj/cam/Blokus/src/agents/cache/minimax"):
        self.name = name
        self.depth = depth
        self.board_size = board_size
        self.env = gym.make(
            'gymnasium_env/Blokus-v0', 
            board_size=board_size, 
            num_players=2, 
            render_mode='console', 
            render_scale=10, 
            disable_env_checker=True, 
            neighborhood_dir="/Users/mario/Documents/proj/cam/Blokus/gymnasium_env/envs/auxiliary/pre_neighbors"
        )
        self.env = self.env.unwrapped
        self.env.order_enforce = False
        self.use_cache = use_cache
        if self.use_cache:
            self.cache_path = f"{cache_dir}/minimax_depth{self.depth}_bz{board_size}.pkl"
            self.load_cache()

    def get_action(self, env, obs):
        encoded_board = encode_board_string(obs["state"])
        if self.use_cache and encoded_board in self.cache:
            return random.choice(self.cache[encoded_board]["actions"]).action_id
        state = env.capture_state()
        self.env.restore_state(state)
        good_actions, best_value = self.minimax(obs, self.depth)
        if self.use_cache:
            self.cache[encoded_board] = {
                "actions": good_actions,
                "value": best_value
            }
        # print(best_value)
        # print(encoded_board)
        return random.choice(good_actions).action_id
    
    def save_cache(self):
        if not self.use_cache:
            return
        if self.cache_path is not None:
            with open(self.cache_path, 'wb') as file:
                pickle.dump(self.cache, file)

    def load_cache(self):
        if self.use_cache and self.cache_path is not None and os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as file:
                self.cache = pickle.load(file)
            print(f"Cache loaded for MiniMaxAgent with depth {self.depth} and board size {self.board_size}")
        else:
            self.cache = {}
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            print(f"Cache not found for MiniMaxAgent with depth {self.depth} and board size {self.board_size}")

    def minimax(self, obs, depth):
        if depth <= 0:
            return None, 0
        actions = np.array([BlokusAction(board_size=self.board_size, action_id=action_id) for action_id in obs["possible_actions"]])
        value = np.zeros(len(actions))
        best_value = -np.inf
        if depth == 1:
            for idx, action in enumerate(actions):
                psz = action.piece.size
                value[idx] = psz
                if psz > best_value:
                    best_value = psz
        else:
            state = self.env.capture_state()
            for idx, action in enumerate(actions):
                psz = action.piece.size
                new_obs, new_reward, term, trunc, _ = self.env.step(action.action_id)
                assert new_reward == psz
                assert not trunc
                if term:
                    value1 = psz
                else:
                    if new_obs["current_player"] == obs["current_player"]: ## your turn again!
                        value1 = self.minimax(new_obs, depth - 2)[1] + psz
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