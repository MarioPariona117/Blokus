import numpy as np
from .agent import Agent

class FunctionAgent(Agent):
    def __init__(self, name="Agent", func=None):
        self.name = name
        self.func = func # (obs, action, next_obs) -> float

    def choose_action(self, obs, actions, next_obss):
        action_values = np.array(np.map(lambda x: self.func(obs, x[0], x[1]), zip(actions, next_obss)))
        return actions[np.argmax(action_values)]
    
class SimpleFunctionAgent(Agent):
    def __init__(self, name="Agent", func=None):
        self.name = name
        self.func = func # (actions) -> action

    def get_action(self, env, obs):
        actions = obs["possible_actions"]
        return self.func(env, actions)