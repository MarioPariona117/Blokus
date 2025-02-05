from typing import Callable, Tuple, List
import gymnasium as gym
from typing import SupportsFloat, SupportsInt
import numpy as np
import random
import threading
import pickle
import time
from gymnasium.core import ObsType, ActType
from tqdm import tqdm
from ..minimax_agent import MiniMaxAgent
from src.utils import encode_board_bytes, decode_board_bytes, time_function
from gymnasium_env.envs.blokus_env import BlokusEnv, BlokusAction, BlokusPieceManager
from .heuristics import my_heu
from .cache_manager import CacheManager
import wandb

class ABPruningAgent(MiniMaxAgent):
    MAX_DEPTH = 100
    def __init__(
        self, 
        board_size : int, 
        name: str = "ABPruning", 
        depth: int = -1, 
        cache_dir: str = "/Users/mario/Documents/proj/cam/Blokus/src/agents/cache/alpha_beta", 
        use_cache: bool = True, 
        sorted_order: Callable[[BlokusEnv, ObsType, BlokusAction], float | Tuple[float, float]] = lambda x: my_heu(*x),
        testing_mode: Tuple[bool, int | None] = (False, None)
    ):
        """
        Initializes the ABPruningAgent.

        :param board_size: The size of the Blokus board.
        :param name: The name of the agent. Default is "ABPruning".
        :param depth: The depth of the search tree for the minimax algorithm. Default is 100.
        :param cache_dir: The directory where the cache will be stored. Default is None.
        :param use_cache: Whether to use caching to store previously computed states. Default is True.
        :param sorted_order: A function to sort the actions based on a heuristic. It takes an environment, an observation, and an action as input. Default is None.

        """
        self.name = name
        assert depth >= -1, "Depth must be greater than or equal to -1"
        self.depth = depth if depth >= 0 else ABPruningAgent.MAX_DEPTH
        self.board_size = board_size
        self.env: BlokusEnv = gym.make('gymnasium_env/Blokus-v0', board_size=board_size, num_players=2, disable_env_checker=True, testing_mode=False)
        self.env = self.env.unwrapped
        self.env.order_enforce = False
        cache_path = f"{cache_dir}/alpha_beta_depth{depth}_bz{board_size}_bytes.pkl"
        self.use_cache = use_cache
        if self.use_cache:
            self.cache_manager = CacheManager(cache_path=cache_path, time_update=120, time_threshold=1200, size_threshold=1e8)
        self.sorted_order = sorted_order
        ##############################
        self.num_pruned = 0
        self.visited_states = 0
        self.log = []
        self.testing_mode = testing_mode[0]
        self.depth_diff_test = testing_mode[1]
        self.counter = 0
        self.if_print = 1
        wandb.init(project="ABPruningAgent", config={"board_size": board_size})

    # @time_function
    def get_action(self, env: BlokusEnv, obs: ObsType) -> ActType:
        encoded_board = encode_board_bytes(obs["state"])
        if self.use_cache:
            best_action = self.cache_manager.retrieve_action(encoded_board)
            if best_action is not None:
                return best_action
        player = obs["current_player"]
        my_points = obs["points"][player]
        opponent_points = obs["points"][3 - player]
        state = env.capture_state()
        self.env.restore_state(state)
        best_action, best_value = self.minimax(obs, self.depth, alpha=-np.inf, beta=opponent_points - my_points + 1)
        return best_action

    def minimax(self, obs: ObsType, depth: int, alpha: float = -np.inf, beta: float = np.inf, theta: float = 0):
        self.visited_states += 1
        if depth <= 0:
            return None, 0

        encoded_board = encode_board_bytes(obs["state"])
        if self.use_cache:
            best_action = self.cache_manager.retrieve_action(encoded_board)
            best_value = self.cache_manager.retrieve_value(encoded_board)
            if best_action is not None:
                return best_action, best_value

        best_value, best_action = -np.inf, -1

        state = self.env.capture_state()
        action_ids = obs["possible_actions"]
        actions = [BlokusAction(board_size=self.board_size, action_id=action_id) for action_id in action_ids]
        sorted_actions: List[BlokusAction] = sorted(actions, key=lambda action: self.sorted_order((self.env, obs, action)))
        if self.testing_mode:
            if depth >= self.depth - self.depth_diff_test:
                sorted_actions = tqdm(sorted_actions)

        for action in sorted_actions:
            if self.testing_mode:
                if depth >= self.depth - self.depth_diff_test:
                    sorted_actions.set_description(f"BValue: {best_value}, BAction: {best_action}, CAction: {action})")
                    # mine(self.env, obs, action, print_=True)
                    sorted_actions.refresh()
                    
            psz = action.piece.size
            new_obs, new_reward, term, trunc, _ = self.env.step(action.action_id)
            assert psz == new_reward
            assert not trunc
            if term:
                value = new_reward
            else:
                if new_obs["current_player"] == obs["current_player"]:
                    _, value = self.minimax(new_obs, depth - 2, alpha - psz, beta - psz)
                else:
                    _, value = self.minimax(new_obs, depth - 1, -beta + psz, -alpha + psz)
                    value = -value
            
                value += psz
            
            if value > best_value:
                best_value = value
                best_action = action

            self.env.restore_state(state)
            alpha = max(alpha, value)
            if alpha >= beta:
                self.num_pruned += 1
                pruned_percentage = (self.num_pruned / self.visited_states) * 100 if self.visited_states > 0 else 0
                if self.visited_states % 1000 == 0:
                    wandb.log({"num_pruned": self.num_pruned, "visited_states": self.visited_states, "pruned_percentage": pruned_percentage})
        
        if self.use_cache:
            self.cache_manager.update_cache(encode_board_bytes(obs["state"]), best_action.action_id, best_value, obs["steps"])

        return best_action.action_id, best_value
    
    def save_cache(self):
        if self.use_cache:
            self.cache_manager.save_cache()
        else:
            raise Exception(f"Cache not used on {self.name} agent.")

# SORTING HEURISTICS #
# 1. piece_size
# 2. maximise_difference of expanders (be sure to keep the 1 for last)
# 3. use locked squares or its difference (just a bit)
# 4. average distance among new expanders (seems powerful, but maybe no)
# 5. guided q-value (maybe)
##
