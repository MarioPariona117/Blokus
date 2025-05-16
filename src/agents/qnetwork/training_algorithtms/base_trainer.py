import random
from typing import Optional

class BaseTrainer:
    def __init__(self, agent, seed: Optional[int] = None):
        self.agent = agent
        self.rng = random.Random(seed)

    def save_model(self, path: str):
        """Saves the model to the specified path."""
        pass
