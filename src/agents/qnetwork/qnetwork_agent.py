from datetime import datetime
import os
import json
from datetime import datetime
from typing import Optional, Type

import torch

from proj_config import MODEL_DIR
from gymnasium_env.envs import BlokusAction, BlokusEnv
from gymnasium.core import ObsType, ActType

from src.agents import Agent
from .archis.base_arch import BaseArch

class QNetworkAgent(Agent):
    def __init__(
        self,
        device: str,
        board_size: int,
        model_class: Type[BaseArch],
        model_folder: Optional[str] = None,
        dir: Optional[str] = None,
    ):
        """
        Initializes the QNetworkAgent.

        Args:
            device (str): Device to run the model on (e.g., 'cpu' or 'cuda').
            board_size (int): Size of the Blokus board.
            model_class (Type[BaseArch]): Neural network architecture class.
            model_folder (Optional[str], optional): Folder name for saving the model. If None, a timestamped folder is generated.
            dir (Optional[str], optional): Full directory path for saving/loading the model. If None, it is constructed automatically.
        """
        super().__init__(name="DQNAgent")
        # Models
        self.board_size = board_size
        self.model_class = model_class
        self.device = torch.device(device)
        self.policy_net = model_class(board_size=self.board_size).to(self.device) # NN model

        # Directory structure: models/dqnagent/unicolor/board_10/2024-05-03_12-00-00/
        if dir is None:
            if model_folder is None:
                model_folder = self.generate_model_folder()

            self.dir = os.path.join(
                MODEL_DIR,
                self.__class__.__name__.lower(),
                self.model_class.__name__.lower(),
                f"board_{self.board_size}",
                model_folder
            )
        else:
            self.dir = dir

        os.makedirs(name=self.dir, exist_ok=True)

        self.save_config()

        self.load_model()
        self.testing = False

    def generate_model_folder(self) -> str:
        """Generates a file name based on the current date and time."""
        now = datetime.now()
        return f"{now.strftime('%Y%m%d_%H%M%S')}"
    
    def get_q_values(self, obs: ObsType) -> torch.Tensor:
        with torch.no_grad():
            encode_actions = self.policy_net.encode(obs, obs["possible_actions"], device=self.device)
            return self.policy_net(encode_actions)
        
    def get_action(self, env: BlokusEnv, obs: ObsType) -> ActType:
        assert self.testing
        q_values = self.get_q_values(obs)
        action_idx = q_values.argmax().item()
        action_id = obs["possible_actions"][action_idx]
        return action_id

    def train(self) -> None:
        """Sets the model to training mode."""
        self.policy_net.train()
        self.testing = False

    def eval(self) -> None:
        """Sets the model to evaluation mode."""
        self.policy_net.eval()
        self.testing = True

    def save_model(self, path: str | None = None) -> None:
        """Saves the model to the given path."""
        path = path or os.path.join(self.dir, "policy_net.pth")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)
        
    def save_config(self) -> None:
        config = {
            "board_size": self.board_size,
            "model_class": self.model_class.__name__,
        }
        config_path = os.path.join(self.dir, "agent_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    def load_model(self, path: str | None = None) -> None:
        """Loads the model from the given path if it exists."""
        path = path or os.path.join(self.dir, "policy_net.pth")
        if os.path.exists(path):
            try:
                self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
                print(f"Model loaded successfully from {path}")
            except Exception as e:
                raise e
                print("Failed to load model:", e)
        else:
            print(f"Model file not found at {path}. Initializing a new model.")
        self.policy_net.to(self.device)