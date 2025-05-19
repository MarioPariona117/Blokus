from typing import Callable, Tuple, List, SupportsFloat, SupportsInt
import gymnasium as gym
from tqdm import tqdm
import numpy as np
import threading
import random
import pickle
import time
import os

from proj_config import CACHE_DIR

from src.utils import encode_board_bytes, decode_board_bytes, time_function
from gymnasium_env import BlokusEnv, BlokusAction, BlokusPieceManager
from ..minimax.minimax_agent import MiniMaxAgent
from gymnasium.core import ObsType, ActType

from .cache_manager import CacheManager
# import wandb

class ABPruningAgent(MiniMaxAgent):
    MAX_DEPTH = 100
    def __init__(
        self, 
        board_size : int, 
        use_cache: bool,
        cache_dir: str = None, 
        name: str = "ABPruning", 
        depth: int = -1, 
        heuristic: Callable[[BlokusEnv, ObsType, BlokusAction], float | Tuple[float, float]] = lambda env, obs, action: 0.0,
        *args, **kwargs
    ):
        """
        Initializes the ABPruningAgent.

        Args:
            use_cache (bool): Whether to use caching to store previously computed states.
            board_size (int): The size of the Blokus board.
            name (str, optional): The name of the agent. Defaults to "ABPruning".
            depth (int, optional): The depth of the search tree for the minimax algorithm. Defaults to -1.
            cache_dir (str, optional): The directory where the cache will be stored. Defaults to None.
            heuristic (Callable[[BlokusEnv, ObsType, BlokusAction], float | Tuple[float, float]], optional): 
            A function to sort the actions based on a heuristic. It takes an environment, an observation, 
            and an action as input. Defaults to my_heu.
            and the depth difference for testing. Defaults to (False, None).
        """
        assert depth >= -1, "Depth must be greater than or equal to -1"
        self.depth = depth if depth >= 0 else ABPruningAgent.MAX_DEPTH
        super().__init__(board_size=board_size, name=name, depth=self.depth, use_cache=use_cache, *args, **kwargs)
        if self.use_cache:
            if cache_dir is None:
                cache_dir = os.path.join(CACHE_DIR, "alpha_beta")
            self.cache_path = os.path.join(cache_dir, f"ab_depth{depth}_bz{self.board_size}_bytes/cache.pkl")

            self.cache_manager = CacheManager(cache_path=self.cache_path, time_update=120, time_threshold=1200, size_threshold=1e8)
        else:
            print("NOTE: Cache not being used.")
        self.heuristic = heuristic
        ##############################
        self.num_pruned = 0
        self.visited_states = 0
        self.log = []
        self.counter = 0
        self.if_print = 1
        # wandb.init(project="ABPruningAgent", config={"board_size": board_size})

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
        # next_obss = self.env.next_obss(action_ids)
        sorted_actions: List[BlokusAction] = sorted(
            actions, 
            key = lambda action: (-self.heuristic(
                env=self.env, 
                obs=obs, 
                action=action, 
            ))
        )
        for action in sorted_actions:
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
                # if self.visited_states % 1000 == 0:
                #     wandb.log({"num_pruned": self.num_pruned, "visited_states": self.visited_states, "pruned_percentage": pruned_percentage})
        
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
