from typing import Tuple

from .blokus_piece import BlokusPieceManager, Vector2

class BlokusAction:
    @staticmethod
    def id_to_tuple(board_size, action_id: int) -> Tuple[int, int, int, int]:
        return (
            action_id // (BlokusPieceManager.PIECE_VARIANTS_COUNT * board_size),
            (action_id // BlokusPieceManager.PIECE_VARIANTS_COUNT) % board_size,
            *BlokusPieceManager.piece_infos[int(action_id % BlokusPieceManager.PIECE_VARIANTS_COUNT)]
        )

    @staticmethod
    def tuple_to_id(board_size, action_tuple: Tuple[int, int, int, int]) -> int:
        return (
            action_tuple[0] * board_size * BlokusPieceManager.PIECE_VARIANTS_COUNT +
            action_tuple[1] * BlokusPieceManager.PIECE_VARIANTS_COUNT +
            BlokusPieceManager.get_piece_idx(action_tuple[2], action_tuple[3])
        )
    def get_board_transformation(self, dx: int, dy: int, expander: Vector2) -> None:
        shape = BlokusPieceManager.get_piece_dimensions(self.shape_id, self.transform)
        original_piece_transform = BlokusPieceManager.pieces[self.shape_id].raw_index[self.transform]
        if dx == -1: ## needs flip in x axis
            self.x = 1 - self.x - shape.x + expander.x
            original_piece_transform ^= 1
        else:
            self.x = self.x + expander[0]
        if dy == -1:
            self.y = 1 - self.y - shape.y + expander.y
            original_piece_transform ^= 2
        else:
            self.y = self.y + expander[1]
        self.transform = BlokusPieceManager.pieces[self.shape_id].indexes[original_piece_transform]
        self._update_id()

    def __init__(self, board_size: int, action_id: int | None = None, action_tuple: Tuple[int, int, int, int] | None = None):
        self.board_size = board_size
        if action_id is not None:
            self.action_id = action_id
            assert action_tuple is None
            self._update_tuple()
        elif action_tuple is not None:
            self.x, self.y, self.shape_id, self.transform = action_tuple
            assert action_id is None
            self._update_id()
        else:
            raise ValueError("Either action_id or action_tuple must be provided")

    @property    
    def action_tuple(self):
        return self.x, self.y, self.shape_id, self.transform
    
    def __str__(self):
        return f"Action(id = {self.action_id} x = {self.x}, y = {self.y}, p = {self.shape_id}, t = {self.transform})"
    
    def update_position(self, dx: int = 0, dy: int = 0):
        self.x += dx
        self.y += dy
        self._update_id()

    def _update_id(self):
        self.action_id = self.tuple_to_id(self.board_size, (self.x, self.y, self.shape_id, self.transform))
        
    def _update_tuple(self):
        self.x, self.y, self.shape_id, self.transform = self.id_to_tuple(self.board_size, self.action_id)