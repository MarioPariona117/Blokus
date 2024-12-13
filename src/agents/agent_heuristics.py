import numpy as np
import random

def greedy(env1, actions):
    def f(x):
        h, w, pid, pt = env1._action_to_tuple(x)
        return len(env1.pieces[pid].transformations[pt].shape)
    values = np.vectorize(f)(actions)
    max_value = np.max(values)
    idx = random.choice(np.where(values == max_value)[0])
    return actions[idx]