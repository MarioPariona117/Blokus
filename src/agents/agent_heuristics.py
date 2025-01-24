import numpy as np
import random
from gymnasium_env.envs import BlokusAction, BlokusEnv, BlokusPieceManager

def greedy(env, obs):
    """Selects action placing the largest piece."""
    def f(action: BlokusAction):
        return -action.piece.size
    
    f_values = np.vectorize(f)(obs["possible_actions"])
    max_value = np.max(f_values)
    idx = random.choice(np.where(f_values == max_value)[0])
    return obs["possible_actions"][idx]