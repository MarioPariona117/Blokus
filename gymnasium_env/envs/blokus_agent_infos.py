from .blokus_piece import BlokusPieceManager, PIECE_SHAPE_IDS, Vector2
from typing import List, Dict

class AgentInfo:
    """Class to store information about a Blokus agent, including the player ID and available pieces."""
    def __init__(self, player_id, available_shapes: List[str]):
        self.points = 0
        self.player_id = player_id
        self.available_shapes = set(list(available_shapes))
        self.expander_squares = {}
        self.locked_squares = set()
        self.possible_actions = None
        self.started = False

class BlokusAgentInfos:
    def __init__(self, num_players: int):
        self.infos = [
            AgentInfo(player_id=i, available_shapes=PIECE_SHAPE_IDS)
            for i in range(0, num_players)
        ]

    def __getitem__(self, player_id) -> AgentInfo:
        return self.infos[player_id - 1]
    
    def __setitem__(self, player_id, value):
        self.infos[player_id - 1] = value