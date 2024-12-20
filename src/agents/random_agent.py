import random
from .agent import Agent

def random_policy(actions):
    if len(actions) == 0:
        return None
    return random.choice(actions)

class RandomAgent(Agent):
    def __init__(self, name="Agent"):
        super().__init__(name, "random", random_policy)

    def get_action(self, env, obs):
        actions = obs["possible_actions"]
        return random.choice(actions)