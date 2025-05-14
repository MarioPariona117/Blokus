import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import time

from gymnasium_env.envs import BlokusAction, BlokusEnv
from gymnasium.core import ObsType

from .base_arch import BaseArch

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResBlock, self).__init__()
        self.downsample = downsample
        stride = 2 if downsample else 1
        
        # Main branch
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut branch
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.relu(x + shortcut)


# TODO
class UnicolorArch(BaseArch):

    def _precompute_action_encodings(self):
        num_actions = self.board_size * self.board_size * 91
        precomputed_encodings = [
            torch.zeros((num_actions, 1, self.board_size, self.board_size)), 
            torch.zeros((num_actions, 2 * self.board_size + 91))
        ]
        action_data = [BlokusAction(board_size=self.board_size, action_id=action_id) for action_id in range(num_actions)]

        with torch.no_grad():
            for i, action in enumerate(action_data):
                action_tensor = torch.zeros(1, self.board_size, self.board_size)
                piece_positions = [(action.x + pos[0], action.y + pos[1]) for pos in action.piece.body]

                stop = False
                for pos in piece_positions:
                    if pos[0] < 0 or pos[0] >= self.board_size or pos[1] < 0 or pos[1] >= self.board_size:
                        stop = True
                        break
                    action_tensor[0, pos[0], pos[1]] = 1.0
                if stop:
                    continue
                action_xs = F.one_hot(torch.tensor([action.x], dtype=torch.int64), num_classes=self.board_size).float()
                action_ys = F.one_hot(torch.tensor([action.y], dtype=torch.int64), num_classes=self.board_size).float()
                action_piece_ids = F.one_hot(torch.tensor([action.piece.idx], dtype=torch.int64), num_classes=91).float()

                action_list = torch.cat([action_xs, action_ys, action_piece_ids], dim=1)
                
                precomputed_encodings[0][i] = action_tensor
                precomputed_encodings[1][i] = action_list

            return precomputed_encodings
    
    def __init__(self, board_size, board_input_channels=4, action_input_channels=1):
        self.board_size = board_size
        self.board_input_channels = board_input_channels
        self.action_input_channels = action_input_channels
        self.channels1 = 256
        self.channels2 = 256
        self.channels3 = 256
        self.channels4 = 256
        self.channels5 = 256

        # Initial Convolution
        self.conv1 = nn.Conv2d(board_input_channels + action_input_channels, self.channels1, kernel_size=3, stride=1, padding=1)

        self.resblock1 = ResBlock(self.channels1, self.channels2, downsample=False)
        self.resblock2 = ResBlock(self.channels2, self.channels3, downsample=False)
        self.resblock3 = ResBlock(self.channels3, self.channels4, downsample=False)
        self.resblock4 = ResBlock(self.channels4, self.channels5, downsample=False)

        self.fc1 = nn.Linear(self.channels4 * self.board_size * self.board_size + 2 * self.board_size + 91, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        print("Precomputing actions...")
        start_time = time.time()
        self.precomputed_encodings = self._precompute_action_encodings()
        end_time = time.time()
        print(f"Precomputing actions done. Time taken: {end_time - start_time:.2f} seconds.")

    # @time_function
    def forward(self, encoded_actions):
        state, action = encoded_actions
        if state.shape[0] == 0:
            raise ValueError("Empty state tensor")
        x = state.view(-1, self.board_input_channels, self.board_size, self.board_size)
        action, action_flat = action
        try:
            x = torch.cat((state, action), dim=1)
        except Exception as e:
            print(f"State shape: {state.shape}")
            print(f"Action shape: {action.shape}")
            raise e
        x = self.conv1(x)
        _, C, H, W = x.shape  
        x = F.layer_norm(x, normalized_shape=x.shape[1:])  # Apply LayerNorm
        x = F.relu(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, action_flat), dim=1)
        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze(1)
        return x
    
    def encode_state(self, obs: ObsType, device='cpu'):
        try:
            current_player = obs["current_player"]
        except Exception as e:
            print("asdasdasdasdasd\n\n\n", obs)
            exit(0)
            raise e
        try:
            board = torch.tensor(obs["state"], dtype=torch.float32)
        except Exception as e:
            print(f"Error encoding board    : {e}")
            print(obs["state"])
            raise e
        expanders = obs["expander_squares"]
        my_cells = (board == current_player).float()  # My cells
        my_expanders = torch.zeros_like(board, dtype=torch.float32)  # Expanders[current_player]
        his_cells = (board == 3 - current_player).float()  # Opponent's cells
        his_expanders = torch.zeros_like(board, dtype=torch.float32)  # Expanders[3 - current_player]

        if current_player in [1, 2]:
            for pos in expanders[current_player]:
                my_expanders[pos[0], pos[1]] = 1.0

            for pos in expanders[3 - current_player]:
                his_expanders[pos[0], pos[1]] = 1.0

        state_tensor = torch.stack([my_cells, my_expanders, his_cells, his_expanders], dim=0)

        return state_tensor.to(device)
    
    def encode_actions(self, action_ids, device='cpu'):
        action_tensors = self.precomputed_encodings[0][action_ids].to(device)
        action_lists = self.precomputed_encodings[1][action_ids].to(device)
        return (action_tensors, action_lists)
    
    def encode(self, obs, actions, device='cpu'):
        encoded_state = self.encode_state(obs, device=device)
        encoded_actions = self.encode_actions(actions, device=device)
        repeated_state = encoded_state.unsqueeze(0).expand(len(actions), -1, -1, -1)
        return (repeated_state, encoded_actions)
    
    def cat(self, list_of_encodings):
        return (
            torch.cat([encoding[0] for encoding in list_of_encodings], dim=0),
            (
                torch.cat([encoding[1][0] for encoding in list_of_encodings], dim=0),
                torch.cat([encoding[1][1] for encoding in list_of_encodings], dim=0)  # Concatenate action encodings
            )
        )