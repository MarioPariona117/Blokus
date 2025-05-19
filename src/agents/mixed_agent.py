from gymnasium.core import ActType, ObsType
import random
from typing import Optional, List, SupportsFloat

from gymnasium_env import BlokusEnv

from .agent import Agent
class MixedAgent(Agent):
    def __init__(self, agent_list: List[Agent], agent_probs: List[SupportsFloat], name: str = "Mixed"):
        """
        Initializes the MixedAgent with a list of agents and their corresponding probabilities.

        Args:
            name (str): Name of the agent.
            agent_list (list): List of agents to choose from.
            agent_probs (list): List of probabilities corresponding to each agent in agent_list.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        super().__init__(name=name)
        assert len(agent_list) == len(agent_probs), "Agent list and probabilities must have the same length"
        assert sum(agent_probs) == 1, "Probabilities must sum to 1"
        self.agent_list = agent_list
        self.agent_probs = agent_probs

    def get_action(self, env: BlokusEnv, obs: ObsType) -> ActType:
        actions = obs["possible_actions"]
        assert len(actions) > 0, "No possible actions available"
        agent = self.rng.choices(self.agent_list, self.agent_probs)[0]
        return agent.get_action(env, obs)