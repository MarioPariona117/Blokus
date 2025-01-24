from typing import Tuple

from .blokus_piece import BlokusPieceManager, BlokusPiece, Vector2

class BlokusAction:
    @staticmethod
    def tuple_to_id(board_size, x: int, y: int, shape_id, transform) -> int:
        return (
            x * board_size * BlokusPieceManager.PIECE_VARIANTS_COUNT +
            y * BlokusPieceManager.PIECE_VARIANTS_COUNT +
            BlokusPieceManager.pieces[shape_id].get_transformation(transform).idx
        )
    
    @staticmethod  
    def transform(board_row, board_col, dx, dy, shape_id, piece_transformation) -> Tuple[int, int, int]:
        h, w = BlokusPieceManager.pieces[shape_id].transformations[piece_transformation].dimensions
        piece_transformation = BlokusPieceManager.pieces[shape_id].raw_index[piece_transformation]
        if dx == -1: 
            board_row = 1 - board_row - h
            piece_transformation ^= 1
        if dy == -1:
            board_col = 1 - board_col - w
            piece_transformation ^= 2
        piece_transformation = BlokusPieceManager.pieces[shape_id].indexes[piece_transformation]
        return board_row, board_col, piece_transformation
    
    # def _transform(self, dx: int, dy: int, expander: Vector2) -> None: # SLOW FOR NO APPARENT REASON
    #     dimensions = self.piece.dimensions
    #     new_transform = self.piece.transform
    #     if dx == -1: ## needs flip in x axis
    #         self.x = 1 - self.x - dimensions.x
    #         new_transform = BlokusPieceManager.flip_x(self.piece.shape_id, new_transform)
    #         # self.piece = BlokusPieceManager.flip_x(self.piece)
    #     if dy == -1:
    #         self.y = 1 - self.y - dimensions.y
    #         new_transform = BlokusPieceManager.flip_y(self.piece.shape_id, new_transform)
    #         # self.piece = BlokusPieceManager.flip_y(self.piece)
    #     self.piece = BlokusPieceManager.get_piece(shape_id=self.piece.shape_id, transform=new_transform)
    #     self.x += expander.x
    #     self.y += expander.y

    def __init__(self, *, board_size: int, action_id: int | None = None, action_tuple: Tuple[int, int, BlokusPiece] | None = None, transform: Tuple[int, int, Vector2] = None) -> None:
        self.board_size = board_size
        if isinstance(action_id, BlokusAction):
            print(action_id)
            raise ValueError("Action id must be an integer")
        
        if action_id is not None:
            self.action_id = action_id
            assert action_tuple is None
            self._update_tuple()
        elif action_tuple is not None:
            self._x, self._y, self.piece = action_tuple
            if not isinstance(self.piece, BlokusPiece):
                raise ValueError("Piece must be a BlokusPiece object")
                # and 0 <= self.piece < BlokusPieceManager.PIECE_VARIANTS_COUNT:
                # self.piece = BlokusPieceManager.get_piece(self.piece)
            assert action_id is None
            if transform:
                self._transform(*transform)
            self._update_id()
        else:
            raise ValueError("Either action_id or action_tuple must be provided")
        # self._update_id()
    
    @property
    def x(self):
        return int(self._x)
    
    @property
    def y(self):
        return int(self._y)

    def create_rot180(self):
        return BlokusAction(
            board_size=self.board_size, 
            action_tuple=(
                self.board_size - self._x - 1,
                self.board_size - self._y - 1, 
                BlokusPieceManager.get_piece(
                    shape_id=self.piece.shape_id,
                    transform=BlokusPieceManager.rot180(self.piece.shape_id, self.piece.transform)
                )
            )
        )
    
    @property    
    def action_tuple(self):
        return self._x, self._y, self.piece
    
    def __str__(self):
        return f"Action(id = {self.action_id} x = {self._x}, y = {self._y}, p = {self.piece})"
    
    def update_position(self, dx: int = 0, dy: int = 0):
        self._x += dx
        self._y += dy
        self._update_id()

    def _update_id(self):
        self.action_id = (
            self._x * self.board_size * BlokusPieceManager.PIECE_VARIANTS_COUNT +
            self._y * BlokusPieceManager.PIECE_VARIANTS_COUNT +
            self.piece.idx
        )

    def _update_tuple(self):
        try:
            self._x = self.action_id // (BlokusPieceManager.PIECE_VARIANTS_COUNT * self.board_size)
            self._y = (self.action_id // BlokusPieceManager.PIECE_VARIANTS_COUNT) % self.board_size
            self.piece = BlokusPieceManager.get_piece(
                piece_id=int(self.action_id % BlokusPieceManager.PIECE_VARIANTS_COUNT)
            )
        except Exception as e:
            print(self.action_id)
            raise e
