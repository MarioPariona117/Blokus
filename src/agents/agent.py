import gymnasium as gym
import random
    
def random_policy(actions):
    if len(actions) == 0:
        return None
    return random.choice(actions)

def maximize_policy(actions, action_to_value):
    if not actions:
        return None
    return max(actions, key=lambda action: action_to_value(action))

class Agent:
    policy_map = {
        "random": random_policy,
        "maximize": maximize_policy,
    }
    def __init__(self, name="Agent", policy=None):
        self.name = name
        self.policy = self.policy_map[policy]  # Function to determine the agent's actions

    def get_action(self, env: gym.Env):
        action = self.policy(env) 
        return action