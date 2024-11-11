import gymnasium as gym
import random
class Agent:
    def __init__(self, name="Agent", policy=None):
        self.name = name
        self.policy = policy  # Function to determine the agent's actions

    def get_action(self, env: gym.Env):
        action = self.policy(env) 
        return action
    
def random_policy(env):
    return random.choice(env.possible_actions_efficient(env.current_player))

def maximize_policy(env, action_value):
    actions = env.action_space.all()
    return max(actions, key=lambda action: action_value(action))

# def action_to_int(action):
#     # Placeholder function, replace with actual logic
#     return int(action)