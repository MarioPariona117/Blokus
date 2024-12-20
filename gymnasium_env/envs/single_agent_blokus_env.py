import gymnasium as gym
from gymnasium_env.envs.blokus_env import BlokusEnv
from src.agents.agent import Agent, RandomAgent, SimpleFunctionAgent, MiniMaxAgent

class SingleAgentBlokusEnv(BlokusEnv):
    HIDDEN_AGENT_INDEX = 2
    def __init__(self, hidden_agents = None, num_players = 2, player_turn = 1, *args, **kwargs, ):
        if not (1 <= player_turn <= num_players):
            raise Exception(f"player_turn must be in the range [1 - {num_players}]")
        super(SingleAgentBlokusEnv, self).__init__(num_players=num_players, *args, **kwargs)
        if hidden_agents is None:
            hidden_agents = [None] + [RandomAgent(name=f"random_hidden_agent_{i}") for i in range(num_players)]
        self.hidden_agents = hidden_agents
        self.player_turn = player_turn
        
    def step(self, action):
        assert self.current_player == 1
        obs, total_reward, terminated, truncated, info = super(SingleAgentBlokusEnv, self).step(action)
        if terminated or truncated: 
            return obs, total_reward, terminated, truncated, info
        
        while self.current_player != self.player_turn:
            assert len(obs["possible_actions"]) > 0
            action = self.hidden_agents[self.current_player].get_action(self, obs)
            obs, reward, terminated, truncated, info = super(SingleAgentBlokusEnv, self).step(action)
            total_reward -= reward / (self.num_players - 1)
            assert not truncated
            if terminated:
                break
        return obs, total_reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        return super(SingleAgentBlokusEnv, self).reset(*args, **kwargs)
    
    def save_caches(self):
        for agent in self.hidden_agents:
            if agent is not None:
                try:
                    agent.save_cache()
                except Exception as e:
                    print(f"Could not save cache for {agent.name}: {e}")    
                    
