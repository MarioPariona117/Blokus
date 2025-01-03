import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym

from agent import Agent

class CustomLossWithReLU(nn.Module):
    def __init__(self, threshold=-50):
        super(CustomLossWithReLU, self).__init__()
        self.threshold = threshold

    def forward(self, predicted_values, target_values):
        # Calculate the absolute difference between predicted and target values
        loss_diff = (predicted_values - target_values).abs()

        # Apply ReLU to mask out invalid actions (those with predicted values <= threshold)
        masked_loss = torch.relu(loss_diff * (predicted_values > self.threshold).float())

        # The total loss is the sum of the individual losses for valid actions
        return masked_loss.mean()
    
class StateToActionsNN(nn.Module):
    def __init__(self, board_size, num_actions=2**16):
        """
        Neural network that predicts values for all actions given a state.
        """
        super(StateToActionsNN, self).__init__()
        self.board_size = board_size
        self.num_actions = num_actions
        self.fc1 = nn.Linear(board_size**2 + 1, 256)  # Flattened board + player turn
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_actions)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_values = self.fc3(x)
        return action_values

# DQN Agent using the StateToActionsNN
class DNN_MiniMax_Agent:
    def __init__(self, board_size=7, num_actions=2**16):
        self.policy = StateToActionsNN(board_size, num_actions)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.env = gym.make("gymnasium_env/Blokus-v0", board_size=board_size, disable_env_checker=True)
        self.env = self.env.unwrapped
        self.env.order_enforce = False
        self.invalid_action_penalty = -50.0
        self.loss_fn = CustomLossWithReLU(threshold=self.invalid_action_penalty)

    def get_action(self, obs) -> int:
        actions = obs["possible_actions"]
        state = torch.tensor(self.encode_state(obs), dtype=torch.float32)  # Encode state
        
        # Get action values from the policy (neural network)
        action_values = self.policy(state)

        # Apply mask to only consider valid actions
        valid_action_values = action_values * torch.tensor(actions, dtype=torch.float32)
        
        # Select the action with the highest value from the valid actions
        best_action_index = valid_action_values.argmax()
        best_action = torch.tensor(actions, dtype=torch.bool).nonzero(as_tuple=True)[0][best_action_index].item()

        return best_action

    def train_step(self, batch, invalid_penalty=-100.0):
        self.policy.train()
        total_loss = 0

        for example in batch:
            state = torch.tensor(example["state"], dtype=torch.float32)
            target_values = torch.tensor(example["target_values"], dtype=torch.float32)
            possible_actions = torch.tensor(example["possible_actions"], dtype=torch.float32)

            # Get the predicted action values
            predicted_values = self.policy(state)

            # Compute the loss
            loss = self.loss_fn(predicted_values, target_values, possible_actions, invalid_penalty)
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(batch)

    def encode_state(self, obs):
        """
        Encodes the state for input into the neural network.
        Args:
            obs: The observation dictionary containing 'state' and 'player_turn'.
        Returns:
            A flattened vector representing the board state plus the player turn.
        """
        board = np.array(obs["state"]).flatten()  # Flatten the board (board_size x board_size)
        player_turn = np.array([obs["player_turn"]])  # Player turn (0 or 1)
        return np.concatenate([board, player_turn])  # Concatenate board + player turn