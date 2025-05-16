from gymnasium.core import ActType, ObsType
import random
from typing import Optional

from gymnasium_env import BlokusEnv

from .agent import Agent
class RandomAgent(Agent):
    def __init__(self, name: str = "RandomAgent", *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def get_action(self, env: BlokusEnv, obs: ObsType) -> ActType:
        actions = obs["possible_actions"]
        assert len(actions) > 0, "No possible actions available"
        return self.rng.choice(actions)