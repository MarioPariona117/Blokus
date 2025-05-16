import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import time

from gymnasium_env.envs import BlokusAction, BlokusEnv, PIECE_SHAPE_IDS
from gymnasium.core import ObsType

from .base_arch import BaseArch

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResBlock, self).__init__()
        self.downsample = downsample
        stride = 2 if downsample else 1
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(out_channels)
        
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
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        # x = self.bn2(x)
        return self.relu(x + shortcut)

class ColorfulArchEmbedding(BaseArch):

    def _precompute_action_encodings(self):
        num_actions = self.board_size * self.board_size * 91
        precomputed_encodings = [
            torch.zeros((num_actions, 1, self.board_size, self.board_size)), 
            torch.zeros((num_actions, 3))
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
                action_list = torch.tensor([action.x * self.board_size + action.y, self.shape_to_int[action.piece.shape_id], action.piece.idx], dtype=torch.float32)
                assert action_list.shape == (3,)
                precomputed_encodings[0][i] = action_tensor
                precomputed_encodings[1][i] = action_list

            return precomputed_encodings
    
    def __init__(self, board_size, board_input_channels=44, action_input_channels=1, embedding_action_dim=24, embedding_pos_dim=16):
        super().__init__()
        self.board_size = board_size
        self.board_input_channels = board_input_channels
        self.action_input_channels = action_input_channels
        self.channels1 = 128
        self.channels2 = 128
        self.channels3 = 128
        self.channels4 = 128
        self.channels5 = 32
        self.action_linear_dim = 256
        # Initial Convolution
        self.conv1 = nn.Conv2d(board_input_channels + action_input_channels, self.channels1, kernel_size=3, stride=1, padding=1)

        self.resblock1 = ResBlock(self.channels1, self.channels2, downsample=False)
        self.resblock2 = ResBlock(self.channels2, self.channels3, downsample=False)
        # self.resblock3 = ResBlock(self.channels3, self.channels4, downsample=False)
        self.resblock4 = ResBlock(self.channels4, self.channels5, downsample=False)

        self.fc1 = nn.Linear(self.channels5 * self.board_size * self.board_size + 42 * embedding_action_dim  + self.action_linear_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        self.embedding_action_dim = embedding_action_dim
        self.embedding_pos_dim = embedding_pos_dim

        self.position_embedding = nn.Embedding(self.board_size * self.board_size, self.embedding_action_dim)
        self.player_embedding = nn.Embedding(2, self.embedding_action_dim)
        self.shape_embedding = nn.Embedding(21, self.embedding_action_dim)
        self.piece_embedding = nn.Embedding(91, self.embedding_action_dim)

        self.action_to_bigger = nn.Linear(self.embedding_action_dim, self.action_linear_dim)
        self.shape_to_int = {
            shape: i for i, shape in enumerate(PIECE_SHAPE_IDS)
        }
        self.precomputed_encodings = self._precompute_action_encodings()

    def forward(self, encoded_actions): # (B, )
        state, action = encoded_actions 
        state, possible_shapes_mask = state
        action, action_flat = action
        batch_size = state.shape[0]

        # state: (B, 44, 10, 10)
        # action: (B, 1, 10, 10)
        # possible_shapes_mask: (B, 42)
        # action_flat: (B, 3)
        if batch_size == 0:
            raise ValueError("Empty state tensor")
            return torch.empty(0, device=x.device, dtype=torch.float32)
        x = state.view(-1, self.board_input_channels, self.board_size, self.board_size)
        x = torch.cat((state, action), dim=1)
        x = self.conv1(x)
        _, C, H, W = x.shape
        # x = F.layer_norm(x, normalized_shape=x.shape[1:])
        x = F.relu(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        # x = self.resblock3(x)
        x = self.resblock4(x)
        x = x.view(batch_size, -1)
        # action_flat = action_flat[:, 0], action_flat[:, 2] (pos, shape_id, piece_id) 
        action_pos_emb = self.position_embedding(action_flat[:, 0].long()) # (B, embedding_action_dim)
        action_shape_emb = self.shape_embedding(action_flat[:, 1].long()) # (B, embedding_action_dim)
        action_piece_emb = self.piece_embedding(action_flat[:, 2].long()) # (B, embedding_action_dim)
        action_flat = action_pos_emb + action_shape_emb + action_piece_emb # (B, embedding_action_dim)

        action_flat = self.action_to_bigger(action_flat) # (B, channels5)

        player_id = torch.cat([torch.zeros(21, device=x.device), torch.ones(21, device=x.device)], dim=0)
        player_id = self.player_embedding(player_id.long()).unsqueeze(0).expand(batch_size, -1, -1)  # (B, 42, embedding_action_dim)
        shape_indices = torch.arange(42, device=x.device).unsqueeze(0).expand(batch_size, -1) % 21 # (B, 42)
        shape_ids = self.shape_embedding(shape_indices) # (B, 42, embedding_action_dim)
        possible_shapes = player_id + shape_ids # (B, 42, embedding_action_dim)
        # print(possible_shapes_mask) # (B, 1, 42)
        # print(possible_shapes.shape, possible_shapes_mask.shape)
        possible_shapes *= possible_shapes_mask.squeeze(1).unsqueeze(2) # (B, 42, embedding_action_dim)
        possible_shapes = possible_shapes.view(batch_size, -1)
        # print(x.shape, possible_shapes.shape, action_flat.shape)
        x = torch.cat((x, possible_shapes, action_flat), dim=1)
        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze(1)
        return x
    
    def encode_state(self, obs: ObsType, device='cpu'):
        current_player = obs["current_player"]
        expanders = obs["expander_squares"]

        my_expanders = torch.zeros((self.board_size, self.board_size), dtype=torch.float32)  # Expanders[current_player]
        his_expanders = torch.zeros((self.board_size, self.board_size), dtype=torch.float32)  # Expanders[3 - current_player]

        encoding = obs["encoding"]

        if current_player in [1, 2]:
            for pos in expanders[current_player]:
                my_expanders[pos[0], pos[1]] = 1.0

            for pos in expanders[3 - current_player]:
                his_expanders[pos[0], pos[1]] = 1.0
        

        state_tensor = torch.cat([encoding, my_expanders.unsqueeze(0), his_expanders.unsqueeze(0)], dim=0)

        my_available_shapes, his_available_shapes = [], []

        if current_player in [1, 2]:
            my_available_shapes = [
                self.shape_to_int[shape] for shape in obs["available_shapes"][current_player]
            ]

            his_available_shapes = [
                self.shape_to_int[shape] for shape in obs["available_shapes"][3 - current_player]
            ]

        my_action_mask = torch.zeros(21, dtype=torch.float32)
        my_action_mask[my_available_shapes] = 1.0

        his_action_mask = torch.zeros(21, dtype=torch.float32)
        his_action_mask[his_available_shapes] = 1.0

        state_list = torch.cat([
            my_action_mask,
            his_action_mask,
        ], dim=0)

        return (state_tensor.to(device), state_list.to(device))
    
    def encode_actions(self, action_ids, device='cpu'):
        action_tensors = self.precomputed_encodings[0][action_ids].to(device)
        action_lists = self.precomputed_encodings[1][action_ids].to(device)
        return (action_tensors, action_lists)
    
    def encode(self, obs, actions, device='cpu'):
        """
        Encodes the states and actions for a batch of observations and actions.
        Args:
            obs: Observation.
            actions: List of list of actions (subset of obs["possible_actions"]).
            device: Device.
        """
        encoded_state1, encoded_state2 = self.encode_state(obs, device=device)
        encoded_actions = self.encode_actions(actions, device=device)
        repeated_state1 = encoded_state1.unsqueeze(0).expand(len(actions), -1, -1, -1)
        repeated_state2 = encoded_state2.unsqueeze(0).expand(len(actions), -1, -1)
        repeated_state = (repeated_state1, repeated_state2)
        return (repeated_state, encoded_actions)
    
    def cat(self, list_of_encodings):
        # Concatenate the state and action tensors along the first dimension
        return (
            (
                torch.cat([encoding[0][0] for encoding in list_of_encodings], dim=0),
                torch.cat([encoding[0][1] for encoding in list_of_encodings], dim=0) 
            ),
            # torch.cat([encoding[0] for encoding in list_of_encodings], dim=0),
            (
                torch.cat([encoding[1][0] for encoding in list_of_encodings], dim=0),
                torch.cat([encoding[1][1] for encoding in list_of_encodings], dim=0) 
            )
        )