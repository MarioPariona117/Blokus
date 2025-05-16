import numpy as np
import random
from typing import Callable, SupportsFloat

from gymnasium.core import ObsType, ActType
from gymnasium_env import BlokusEnv, BlokusAction
from src.agents.agent import Agent 

class HeuristicAgent(Agent):
    def __init__(
            self, 
            func: Callable[
                [BlokusEnv, ObsType, ActType], SupportsFloat
            ],
            board_size: int,
            name: str = "HeuristicAgent", 
            *args,
            **kwargs
        ):
        
        self.board_size = board_size
        self.env = BlokusEnv(board_size=board_size, num_players=2, testing_mode=False)
        self.func = func # : (env, obs, action) -> float
        super().__init__(name=name, *args, **kwargs)

    def get_action(self, env, obs):
        self.env.restore_state(env.capture_state())
        action_ids = np.array(obs["possible_actions"])
        actions = [
            BlokusAction(board_size=self.board_size, action_id=action_id) for action_id in action_ids
        ]
        action_values = np.array([
            np.atleast_1d(self.func(self.env, obs, action)) for action in actions
        ])
        sorted_indices = np.lexsort(action_values[:, ::-1].T)
        max_idx = sorted_indices[-1]
        max_value = action_values[max_idx]
        try:
            best_action_ids = action_ids[(action_values == max_value).all(axis=1)]
        except Exception as e:
            print(f"Error in action_values: {action_values}")
            print(f"Error in action_ids: {action_ids.shape}")
            print(max_value)
            raise e

        action = self.rng.choice(best_action_ids)
        return action