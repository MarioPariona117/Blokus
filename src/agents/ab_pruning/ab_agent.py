from typing import Callable, Tuple
import gymnasium as gym
import numpy as np
import random
import threading
import pickle
import time
from gymnasium.core import ObsType, ActType
from tqdm import tqdm
from ..minimax_agent import MiniMaxAgent
from src.utils import encode_board_string, decode_board_string, time_function
from gymnasium_env.envs.blokus_env import BlokusEnv
from .heuristics import my_heu
from .cache_manager import CacheManager

class ABPruningAgent(MiniMaxAgent):
    MAX_DEPTH = 100
    def __init__(
        self, 
        board_size : int, 
        name: str = "ABPruning", 
        depth: int = -1, 
        cache_dir: str = "/Users/mario/Documents/proj/cam/Blokus/src/agents/cache/alpha_beta", 
        use_cache: bool = True, 
        sorted_order: Callable[[BlokusEnv, ObsType, ActType, int], float | Tuple[float, float]] = lambda x: my_heu(*x),
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
        self.env = gym.make('gymnasium_env/Blokus-v0', board_size=board_size, num_players=2, disable_env_checker=True, testing_mode=False)
        self.env = self.env.unwrapped
        self.env.order_enforce = False
        cache_path = f"{cache_dir}/alpha_beta_depth{depth}_bz{board_size}.pkl"
        self.use_cache = use_cache
        if self.use_cache:
            self.cache_manager = CacheManager(cache_path=cache_path, time_update=120, time_threshold=480, size_threshold=1e8)
        self.sorted_order = sorted_order
        ##############################
        self.num_pruned = 0
        self.visited_states = 0
        self.log = []
        self.testing_mode = testing_mode[0]
        self.depth_diff_test = testing_mode[1]
        print(self.depth_diff_test)
        self.counter = 0
        self.if_print = 1

    # @time_function
    def get_action(self, env, obs):
        encoded_board = encode_board_string(obs["state"])
        if self.use_cache:
            best_action = self.cache_manager.retrieve_action(encoded_board)
            if best_action is not None:
                return best_action
        player = obs["current_player"]
        my_points = obs["points"][player]
        opponent_points = obs["points"][3 - player]
        state = env.capture_state()
        self.env.restore_state(state)
        best_action, best_value = self.minimax(obs, self.depth, alpha=-np.inf, beta=opponent_points - my_points + 1, theta=opponent_points - my_points)
        # print(best_value, self.env._action_to_tuple(best_action))
        return best_action

    def minimax(self, obs, depth, alpha=-np.inf, beta=np.inf, theta=0):
        self.visited_states += 1
        if depth <= 0:
            return None, 0

        encoded_board = encode_board_string(obs["state"])
        if self.use_cache:
            if self.counter == self.if_print:
                print(self.counter)
                self.if_print *= 2
            self.counter += 1
            best_action = self.cache_manager.retrieve_action(encoded_board)
            best_value = self.cache_manager.retrieve_value(encoded_board)
            if best_action is not None:
                return best_action, best_value

        best_value, best_action = -np.inf, -1
        stopped = False

        state = self.env.capture_state()
        actions = obs["possible_actions"]
        # for action in actions:
        #     print(self.env._action_to_tuple(action))
        # exit(0)
        sorted_actions = sorted(actions, key=lambda action: self.sorted_order((self.env, obs, action, obs["steps"])))
        if self.testing_mode:
            if depth >= self.depth - self.depth_diff_test:
                sorted_actions = tqdm(sorted_actions)
        for action in sorted_actions:
            if self.testing_mode:
                if depth >= self.depth - self.depth_diff_test:
                    sorted_actions.set_description(f"BValue: {best_value}, BAction: {self.env._action_to_tuple(best_action)}, CAction: {self.env._action_to_tuple(action)})")
                    # mine(self.env, obs, action, print_=True)
                    sorted_actions.refresh()
            r, c, pid, pt = self.env._action_to_tuple(action)
            psz = len(self.env.pieces[pid].transformations[pt].shape)
            new_obs, new_reward, term, trunc, _ = self.env.step(action)
            assert psz == new_reward
            assert not trunc
            if term:
                value = new_reward
            else:
                if new_obs["current_player"] == obs["current_player"]:
                    _, value = self.minimax(new_obs, depth - 2, alpha - psz, beta - psz, theta - psz)
                else:
                    _, value = self.minimax(new_obs, depth - 1, -beta + psz, -alpha + psz, -theta + psz)
                    value = -value
            
                value += psz
            
            if value > best_value:
                best_value = value
                best_action = action

            alpha = max(alpha, value)
            if alpha >= beta and best_value >= theta:
                stopped = True
                # print("Pruning")
                self.num_pruned += 1
                pruned_percentage = (self.num_pruned / self.visited_states) * 100 if self.visited_states > 0 else 0
                # print(f"Pruned {self.num_pruned} states out of {self.visited_states} ({pruned_percentage:.2f}%)")
                self.log.append((self.num_pruned, self.visited_states, pruned_percentage))
            self.env.restore_state(state)
            if stopped:
                break
        if best_value > theta or not stopped:
            if self.use_cache:
                self.cache_manager.update_cache(encode_board_string(obs["state"]), best_action, best_value)
        return best_action, best_value
    
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
