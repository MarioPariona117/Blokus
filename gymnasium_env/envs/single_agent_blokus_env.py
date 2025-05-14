import gymnasium as gym
from gymnasium.core import ActType
from typing import SupportsInt
from gymnasium_env.envs.blokus_env import BlokusEnv

class SingleAgentBlokusEnv:
    def __init__(self, base_env: BlokusEnv, player_turn: int, hidden_agents=None):
        self.base_env = base_env
        self.player_turn = player_turn
        self.hidden_agents = [None] + (hidden_agents or [])

        if not (1 <= self.player_turn <= self.base_env.num_players):
            raise Exception(f"player_turn must be in the range [1 - {self.base_env.num_players}]")

    def step(self, action_id: ActType):
        assert isinstance(action_id, SupportsInt), f"action_id must be an integer, got {action_id, type(action_id)}"
        assert self.base_env.current_player == self.player_turn, (
            f"current_player {self.base_env.current_player} must be the same as player_turn {self.player_turn}"
        )

        obs, total_reward, terminated, truncated, info = self.base_env.step(action_id)
        if terminated or truncated:
            return obs, total_reward, terminated, truncated, info

        return self._process_hidden_agents(obs, total_reward, info)

    def reset(self, *args, **kwargs):
        obs, info = self.base_env.reset(*args, **kwargs)
        obs, total_reward, terminated, truncated, info = self._process_hidden_agents(obs, 0, info)
        return obs, info

    def _process_hidden_agents(self, obs, total_reward, info):
        terminated, truncated = False, False
        while self.base_env.current_player != self.player_turn:
            assert len(obs["possible_actions"]) > 0
            action_id = self.hidden_agents[self.base_env.current_player].get_action(self, obs)
            obs, reward, terminated, truncated, info = self.base_env.step(action_id)
            total_reward -= reward
            assert not truncated
            if terminated:
                break
        return obs, total_reward, terminated, truncated, info

    def save_caches(self):
        for agent in self.hidden_agents:
            if agent is not None:
                try:
                    agent.save_cache()
                except Exception as e:
                    print(f"Could not save cache for {agent.name}: {e}")

    def __getattr__(self, name):
        return getattr(self.base_env, name)