from gymnasium.core import ObsType, ActType
from typing import Callable, Tuple, List, SupportsFloat
import numpy as np

from gymnasium_env.envs.blokus_env import BlokusEnv, BlokusAction, BlokusPieceManager

## SIPLE HEURISTICS

_levels = [[] for _ in range(40)]

_levels[7] = ["F", "X", "W", "Y", "N", "T5", "Z5", "V5", "L5", "P", "I5", "U", "T4", "Z4", "L4", "I4", "O", "V3", "I3", "2", "1"]

_levels[5] = ["L5", "F", "X", "W", "Y", "N", "T5", "Z5", "V5", "P", "I5", "U", "T4", "Z4", "L4", "I4", "O", "V3", "I3", "2", "1"]

for l in range(40):
    _levels[l] = {
        shape: rank for rank, shape in enumerate(_levels[l])
    }

def level_7_(obs: ObsType, action: BlokusAction) -> int:
    return -_levels[7][action.piece.shape_id]

def level_5_(obs: ObsType, action: BlokusAction) -> int:
    return -_levels[5][action.piece.shape_id]

def piece_size_(obs: ObsType, action: BlokusAction) -> int:
    psz = action.piece.size
    return psz

# COMPLEX HEURISTICS
def max_our_expanders_diff_(obs: ObsType, action: BlokusAction, next_obs: ObsType, wa: float, wb: float) -> float:
    player = obs["current_player"]
    a = (len(next_obs["expander_squares"][player])) # our expanders
    b = (len(next_obs["expander_squares"][3 - player])) # his expanders
    value = (a * wa + b * wb)
    return value

def max_my_expanders_(obs: ObsType, action: BlokusAction, next_obs: ObsType) -> float:
    return max_our_expanders_diff_(obs, action, next_obs, wa=1, wb=0)

def min_his_expanders_(obs: ObsType, action: BlokusAction, next_obs: ObsType) -> float:
    return max_our_expanders_diff_(obs, action, next_obs, wa=0, wb=-1)

def max_our_expander_diff_1_3_(obs: ObsType, action: BlokusAction, next_obs: ObsType) -> float:
    return max_our_expanders_diff_(obs, action, next_obs, wa=1, wb=3)

def div(a, b):
    return a / b if b != 0 else 100

def mine_(obs: ObsType, action: BlokusAction, next_obs: ObsType) -> float:
    player = obs["current_player"]
    a = len(next_obs["expander_squares"][player])  # our expanders
    b = len(next_obs["expander_squares"][3 - player])  # his expanders
    value = div(a, b) - div(b, a) - (b - a) * 0.22
    return -value

def mine2_(obs: ObsType, action: BlokusAction, next_obs: ObsType) -> float:
    player = obs["current_player"]
    a = len(next_obs["expander_squares"][player])  # our expanders
    b = len(next_obs["expander_squares"][3 - player])  # his expanders
    value = (div(a, b) + div(b, a)) * abs(a - b)
    return value

def min_his_possible_actions_(obs: ObsType, action: BlokusAction, next_obs: ObsType) -> float:
    if obs["current_player"] == next_obs["current_player"]:
        return len(next_obs["possible_actions"])
    else:
        return -len(next_obs["possible_actions"])

def min_our_locked_(obs, action):
    pass

def max_his_locked_(obs, action):
    pass

def min_our_locked_diff_(obs, action):
    pass

def average_distance_among_new_expanders_(obs, action):
    # just do bfs between each pair of new expanders, computing distance for his (not locked)
    pass

def guided_q_value_(obs, action):
    pass

def compile_complex(func: Callable[[ObsType, BlokusAction, ObsType], SupportsFloat]) -> Callable[[BlokusEnv, ObsType, BlokusAction], float]:
    def wrapper(env: BlokusEnv, obs: ObsType, action: BlokusAction) -> float:
        next_obs = build(env, action)
        return func(obs, action, next_obs)
    return wrapper

def compile_simple(func: Callable[[ObsType, BlokusAction], SupportsFloat]) -> Callable[[BlokusEnv, ObsType, BlokusAction], float]:
    def wrapper(env: BlokusEnv, obs: ObsType, action: BlokusAction) -> float:
        return func(obs, action)
    return wrapper

def build(env: BlokusEnv, action: BlokusAction):
    with env.step_context(action.action_id) as result:
        return result[0]

