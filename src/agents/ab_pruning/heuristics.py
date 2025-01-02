from gymnasium_env.envs.blokus_env import BlokusEnv

levels = {
    "F" : 0,
    "X": 1,
    "W": 2,
    "Y": 3,
    "N": 4,
    "T5": 5,
    "L5": 6,
    "Z5": 7,
    "V5": 8,
    "I5": 9,
    "P": 10,
    "U": 11,
    "L4": 12,
    "T4": 13,
    "Z4": 14,
    "I4": 15,
    "O": 16,
    "V3": 17,
    "I3": 18,
    "2": 19,
    "1": 20
}

def piece_size(env, action):
    r, c, pid, pt = env._action_to_tuple(action)
    psz = len(env.pieces[pid].transformations[pt].shape)
    return -psz

def maximise_my_expanders(env: BlokusEnv, obs, action):
    state = env.capture_state()
    player = obs["current_player"]
    new_obs, reward, term, trunc, info = env.step(action)
    new_player = new_obs["current_player"]
    value = len(new_obs["expander_squares"][player != new_player])
    env.restore_state(state)
    return -value

def mine(env, obs, action, print_=False):
    # xd = maximise_my_expanders(env, action)
    # return (piece_size(env, action), maximise_our_expanders_difference(env, obs, action, print_), levels[env._action_to_tuple(action)[2]])
    return (piece_size(env, action))

# def minimise_opponent_expanders(env, action):
    # state = env.capture_state()
    # obs = env.get_all_obs_two_players()
    # player = obs["current_player"]
    # _, reward, term, trunc, info = env.step(action)
    # new_obs = env.get_all_obs_two_players()
    # new_player = new_obs["current_player"]
    # value = len(new_obs["expander_squares"][player == new_player])
    # env.restore_state(state)
    # return value
    # pass
def div(a, b):
    return a / b if b != 0 else 100

def maximise_our_expanders_difference(env: BlokusEnv, obs, action, print_=False):
    state = env.capture_state()
    new_obs, reward, term, trunc, info = env.step(action)
    player = obs["current_player"]
    a = (len(new_obs["expander_squares"][player])) # our expanders
    b = (len(new_obs["expander_squares"][3 - player])) # opponent expanders
    # if a == 5 and b == 2:
    #     print("Heyyy, we got a 5 and 2")
    value = (a - b * 2)
    # value = div(a, b) - div(b, a) - (b - a) * 0.22
    # value = (div(a, b) + div(b, a)) * (a - b)
    # value = (div(a, b) + div(b, a)) * (a - b)
    if print_:
        print(f"Value: {value}, A: {a}, B: {b}")
    # value += random.uniform(-0.5, 0.5)
    env.restore_state(state)
    return -value
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