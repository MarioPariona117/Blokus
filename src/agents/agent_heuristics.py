import numpy as np
import random

def greedy(env1, obs):
    """Selects action placing the largest piece."""
    def f(x):
        h, w, pid, pt = env1._action_to_tuple(x)
        return len(env1.pieces[pid].transformations[pt].shape)
    f_values = np.vectorize(f)(obs["possible_actions"])
    max_value = np.max(f_values)
    idx = random.choice(np.where(f_values == max_value)[0])
    return obs["possible_actions"][idx]