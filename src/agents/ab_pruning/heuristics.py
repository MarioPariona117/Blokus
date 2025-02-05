from gymnasium_env.envs.blokus_env import BlokusEnv
from ...utils import env_action_context
from gymnasium_env.envs.blokus_env import ObsType, BlokusAction, BlokusAction, BlokusPieceManager

def div(a, b):
    return a / b if b != 0 else 100

levels = [[] for _ in range(40)]

levels[7] = ["F", "X", "W", "Y", "N", "T5", "Z5", "V5", "L5", "P", "I5", "U", "T4", "Z4", "L4", "I4", "O", "V3", "I3", "2", "1"]

levels[5] = ["L5", "F", "X", "W", "Y", "N", "T5", "Z5", "V5", "P", "I5", "U", "T4", "Z4", "L4", "I4", "O", "V3", "I3", "2", "1"]

for l in range(40):
    cnt = 0
    d = {}
    for i in levels[l]:
        d[i] = cnt
        cnt += 1
    levels[l] = d

def level_7(env, obs: ObsType, action: BlokusAction) -> int:
    return levels[7][action.piece.shape_id]

def level_5(env, action: BlokusAction) -> int:
    return levels[5][action.piece.shape_id]

def piece_size(env: BlokusEnv, obs: ObsType, action: BlokusAction) -> int:
    psz = action.piece.size
    return -psz

def maximise_our_expanders_difference(env: BlokusEnv, obs: ObsType, action: BlokusAction, wa: float = 1, wb: float = 3, print_: bool = False) -> float:
    with env_action_context(env, action) as new_obs:
        return maximise_our_expanders_difference_(obs, new_obs, action, wa, wb, print_)

def maximise_our_expanders_difference_(obs: ObsType, new_obs: ObsType, action: BlokusAction, wa: float = 1, wb: float = 3, print_: bool = False) -> float:
    player = obs["current_player"]
    a = (len(new_obs["expander_squares"][player])) # our expanders
    b = (len(new_obs["expander_squares"][3 - player])) # opponent expanders
    value = (a * wa - b * wb)
    
    if print_:
        print(f"Value: {value}, Our Expanders: {a}, Opponent Expanders: {b}, Weight A: {wa}, Weight B: {wb}")
        print(f"Value: {value}, A: {a}, B: {b}")

    return -value

def maximise_my_expanders(env: BlokusEnv, obs: ObsType, action: BlokusAction) -> float:
    return maximise_our_expanders_difference(env, obs, action, wa = 1, wb = 0)

def maximise_my_expanders_(obs: ObsType, new_obs: ObsType, action: BlokusAction) -> float:
    return maximise_our_expanders_difference_(obs, new_obs, action, wa=1, wb=0)

def minimise_opponent_expanders(env: BlokusEnv, obs: ObsType, action: BlokusAction) -> float:
    return maximise_our_expanders_difference(env, obs, action, wa=0, wb=1)

def minimise_opponent_expanders_(obs: ObsType, new_obs: ObsType, action: BlokusAction) -> float:
    return maximise_our_expanders_difference_(obs, new_obs, action, wa=0, wb=1)

def my_heu(env: BlokusEnv, obs: ObsType, action: BlokusAction, print_: bool = False) -> float:
    # current_depth = obs["steps"]
    
    # if current_depth < 3:
    #     return level(env, obs, action)
    
    with env_action_context(env, action) as new_obs:
        return (
            mine_(obs, new_obs, action) * mine2_(obs, new_obs, action),
            # level(env, obs, action)
        )
    
def mine_(obs: ObsType, new_obs: ObsType, action: BlokusAction) -> float:
    player = obs["current_player"]
    a = len(new_obs["expander_squares"][player])  # our expanders
    b = len(new_obs["expander_squares"][3 - player])  # opponent expanders
    value = div(a, b) - div(b, a) - (b - a) * 0.22
    # print(a, b, -value)
    return -value

def mine2_(obs: ObsType, new_obs: ObsType, action: BlokusAction) -> float:
    player = obs["current_player"]
    a = len(new_obs["expander_squares"][player])  # our expanders
    b = len(new_obs["expander_squares"][3 - player])  # opponent expanders
    value = (div(a, b) + div(b, a)) * abs(a - b)
    return value

# def # value = div(a, b) - div(b, a) - (b - a) * 0.22
    # value = (div(a, b) + div(b, a)) * (a - b)
    # value = (div(a, b) + div(b, a)) * (a - b)

def minimise_our_locked(env, obs, action):
    pass

def maximise_opponent_locked(env, obs, action):
    pass

def minimise_our_locked_difference(env, obs, action):
    pass

def average_distance_among_new_expanders(env, obs, action):
    # just do bfs between each pair of new expanders, computing distance for opponent (not locked)
    pass

def guided_q_value(env, obs, action):
    pass
