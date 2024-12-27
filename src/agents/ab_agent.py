import numpy as np
import gymnasium as gym
import random
from .minimax_agent import MiniMaxAgent
from src.utils import encode_board_string, decode_board_string
from tqdm import tqdm
from gymnasium_env.envs.blokus_env import BlokusEnv

class ABPruningAgent(MiniMaxAgent):
    def __init__(self, board_size, name="ABPruning", depth=100, cache_dir = "/Users/mario/Documents/proj/cam/Blokus/src/agents/cache/ab_pruning", use_cache=True):
        self.name = name
        assert depth >= -1, "Depth must be greater than or equal to -1"
        self.depth = depth if depth >= 0 else 10000
        self.board_size = board_size
        self.env = gym.make('gymnasium_env/Blokus-v0', board_size=board_size, num_players=2, render_mode='console', render_scale=10, disable_env_checker=True, neighborhood_dir="/Users/mario/Documents/proj/cam/Blokus/gymnasium_env/envs/auxiliary/pre_neighbors", testing_mode=False)
        self.env = self.env.unwrapped
        self.env.order_enforce = False
        self.use_cache = use_cache
        if self.use_cache:
            self.cache_path = f"{cache_dir}/alpha_beta_depth{depth}_bz{board_size}.pkl"
            self.load_cache()
        self.sorted_order = lambda x: mine(self.env, x[0], x[1])
        self.need_step = True
        ##############################
        self.num_pruned = 0
        self.visited_states = 0
        self.log = []

    def get_action(self, env, obs):
        encoded_board = encode_board_string(obs["state"])
        if self.use_cache and encoded_board in self.cache:
            return self.cache[encoded_board]["action"]
        state = env.capture_state()
        self.env.restore_state(state)
        best_action, best_value = self.minimax(obs, self.depth)
        # self.cache[encoded_board] = {
        #     "action": best_action,
        #     "value": best_value
        # }
        print(best_value)
        print(encoded_board)
        return best_action

    def minimax(self, obs, depth, alpha=-np.inf, beta=np.inf):
        self.visited_states += 1
        if depth <= 0:
            return None, 0

        best_value, best_action = -np.inf, -1
        stopped = False

        state = self.env.capture_state()
        sorted_actions = sorted(obs["possible_actions"], key=lambda x: self.sorted_order((obs, x)))
        if depth == self.depth:
            sorted_actions = tqdm(sorted_actions)
        for action in sorted_actions:
            if depth == self.depth:
                sorted_actions.set_description(f"BValue: {best_value}, BAction: {self.env._action_to_tuple(best_action)}, CAction: {self.env._action_to_tuple(action)},")
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
                    _, value = self.minimax(new_obs, depth - 2, alpha - psz, beta - psz)
                else:
                    _, value = self.minimax(new_obs, depth - 1, -beta + psz, -alpha + psz)
                    value = -value
            
                value += psz
            
            if value > best_value:
                best_value = value
                best_action = action

            alpha = max(alpha, value)
            if alpha >= beta:
                stopped = True
                # print("Pruning")
                self.num_pruned += 1
                pruned_percentage = (self.num_pruned / self.visited_states) * 100 if self.visited_states > 0 else 0
                # print(f"Pruned {self.num_pruned} states out of {self.visited_states} ({pruned_percentage:.2f}%)")
                self.log.append((self.num_pruned, self.visited_states, pruned_percentage))
                break
            
            self.env.restore_state(state)
        if not stopped:
            if self.use_cache:
                self.cache[encode_board_string(obs["state"])] = {
                    "action": best_action,
                    "value": best_value
                }
        return best_action, best_value
    
# SORTING HEURISTICS #
# 1. piece_size
# 2. maximise_difference of expanders (be sure to keep the 1 for last)
# 3. use locked squares or its difference (just a bit)
# 4. average distance among new expanders (seems powerful, but maybe no)
# 5. guided q-value (maybe)
##

def piece_size(env, action):
    r, c, pid, pt = env._action_to_tuple(action)
    psz = len(env.pieces[pid].transformations[pt].shape)
    return -psz

def maximise_my_expanders(env: BlokusEnv, obs, action):
    state = env.capture_state()
    player = obs["current_player"]
    new_obs, reward, term, trunc, info = env.step(action)
    # new_obs = env.get_all_obs_two_players()
    new_player = new_obs["current_player"]
    value = len(new_obs["expander_squares"][player != new_player]) - len(obs["expander_squares"][player != new_player])
    env.restore_state(state)
    return -value

def mine(env, obs, action):
    # xd = maximise_my_expanders(env, action)
    return (piece_size(env, action), maximise_my_expanders(env, obs, action))

def minimise_opponent_expanders(env, action):
    state = env.capture_state()
    obs = env.get_all_obs_two_players()
    player = obs["current_player"]
    _, reward, term, trunc, info = env.step(action)
    new_obs = env.get_all_obs_two_players()
    new_player = new_obs["current_player"]
    value = len(new_obs["expander_squares"][player == new_player]) - len(obs["expander_squares"][player == new_player])
    env.restore_state(state)
    return value
    pass

def maximise_our_expanders_difference(env, action):
    state = env.capture_state()
    obs = env.get_all_obs_two_players()
    player = obs["current_player"]
    _, reward, term, trunc, info = env.step(action)
    new_obs = env.get_all_obs_two_players()
    new_player = new_obs["current_player"]
    value = len(new_obs["expander_squares"][player == new_player]) - len(obs["expander_squares"][player == new_player])
    value -= 2 * len(new_obs["expander_squares"][player != new_player]) - 2 * len(obs["expander_squares"][player != new_player])
    value += random.uniform(-0.5, 0.5)
    env.restore_state(state)
    return value
    pass

def minimise_our_locked(env, action):
    pass

def maximise_opponent_locked(env, action):
    pass

def minimise_our_locked_difference(env, action):
    pass

def average_distance_among_new_expanders(env, action):
    # just do bfs between each pair of new expanders, computing distance for opponent (not locked)
    pass

def guided_q_value(env, action):
    pass