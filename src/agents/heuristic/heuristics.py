from gymnasium.core import ObsType, ActType
from typing import Callable, Tuple, List, SupportsFloat
import numpy as np

from gymnasium_env.envs.blokus_env import BlokusEnv, BlokusAction, BlokusPieceManager

from ._heuristics import build, compile_simple, compile_complex
from ._heuristics import (
    level_7_,
    level_5_,
    piece_size_,
    max_my_expanders_,
    min_his_expanders_,
    max_our_expander_diff_1_3_,
    min_his_possible_actions_,
    mine_,
    mine2_
)

level_7 = compile_simple(level_7_)
level_5 = compile_simple(level_5_)
greedy = compile_simple(piece_size_)

max_my_expanders = compile_complex(max_my_expanders_)
min_his_expanders = compile_complex(min_his_expanders_)
max_our_expander_diff_1_3 = compile_complex(max_our_expander_diff_1_3_)
# max_possible_actions = compile_complex(max_possible_actions_)
min_his_possible_actions = compile_complex(min_his_possible_actions_)

mine = compile_complex(mine_)
mine2 = compile_complex(mine2_)

def my_heu(env, obs: ObsType, action: BlokusAction, print_: bool = False) -> float:
    current_depth = obs["steps"]
    
    if current_depth < 3:
        return level_7_(obs, action)
    next_obs = build(env, action)
    return (
        mine_(obs, action, next_obs) * mine2_(obs, action, next_obs)
        # level(obs, action)
    )

def min_actions_after_size(env, obs: ObsType, action: BlokusAction) -> float:
    # pieces_sizes = piece_size_(obs, action)
    # current_depth = obs["steps"]
    # if current_depth < 3:
    #     return level_7_(obs, action)
    next_obs = build(env, action)
    return (
        piece_size_(obs, action),
        min_his_possible_actions_(obs, action, next_obs)
    )