from collections import namedtuple
from typing import List, Tuple, Dict
import numpy as np

__all__ = [
    'BlokusPiece',
    'BlokusPieceTransformations',
    'BlokusPieceManager', 
    'PIECE_SHAPE_IDS'
]

Vector2 = namedtuple('Vector2', ['x', 'y'])

SAMPLE_SHAPE = {
    '1': [Vector2(0, 0)],
    '2': [Vector2(0, 0), Vector2(0, 1)],
    'I3': [Vector2(0, 0), Vector2(0, 1), Vector2(0, 2)],
    'V3': [Vector2(0, 0), Vector2(0, 1), Vector2(1, 1)],
    'I4': [Vector2(0, 0), Vector2(0, 1), Vector2(0, 2), Vector2(0, 3)],
    'L4': [Vector2(0, 0), Vector2(1, 0), Vector2(1, 1), Vector2(1, 2)],
    'O': [Vector2(0, 0), Vector2(0, 1), Vector2(1, 0), Vector2(1, 1)],
    'T4': [Vector2(0, 0), Vector2(0, 1), Vector2(0, 2), Vector2(1, 1)],
    'Z4': [Vector2(0, 0), Vector2(0, 1), Vector2(1, 1), Vector2(1, 2)],
    'F': [Vector2(0, 0), Vector2(0, 1), Vector2(1, 1), Vector2(1, 2), Vector2(2, 1)],
    'I5': [Vector2(0, 0), Vector2(0, 1), Vector2(0, 2), Vector2(0, 3), Vector2(0, 4)],
    'L5': [Vector2(0, 0), Vector2(1, 0), Vector2(2, 0), Vector2(3, 0), Vector2(3, 1)],
    'N': [Vector2(0, 0), Vector2(0, 1), Vector2(1, 1), Vector2(1, 2), Vector2(1, 3)],
    'P': [Vector2(0, 0), Vector2(0, 1), Vector2(1, 0), Vector2(1, 1), Vector2(1, 2)],
    'T5': [Vector2(0, 0), Vector2(0, 1), Vector2(0, 2), Vector2(1, 1), Vector2(2, 1)],
    'U': [Vector2(0, 0), Vector2(0, 2), Vector2(1, 0), Vector2(1, 1), Vector2(1, 2)],
    'V5': [Vector2(0, 0), Vector2(1, 0), Vector2(2, 0), Vector2(2, 1), Vector2(2, 2)],
    'W': [Vector2(0, 0), Vector2(1, 0), Vector2(1, 1), Vector2(2, 1), Vector2(2, 2)],
    'X': [Vector2(0, 1), Vector2(1, 0), Vector2(1, 1), Vector2(1, 2), Vector2(2, 1)],
    'Y': [Vector2(0, 0), Vector2(1, 0), Vector2(2, 0), Vector2(3, 0), Vector2(2, 1)],
    'Z5': [Vector2(0, 0), Vector2(0, 1), Vector2(1, 1), Vector2(2, 1), Vector2(2, 2)],
}

PIECE_SHAPE_IDS = list(SAMPLE_SHAPE.keys())

class BlokusPiece:
    def __init__(
            self,
            shape_id: str,
            body: List[Vector2],
            dimensions: Vector2,
            expanders: Dict[Vector2, List[Vector2]],
            locked: List[Vector2],
            print_scale: int = 2
        ):
        self._idx = None ### from 0-90
        self._shape_id = shape_id
        self._dimensions = dimensions
        self._body = body
        self._expanders = expanders
        self._locked = locked
        self._print_scale = print_scale

    def __str__(self) -> str:
        return f"Piece(id={self._shape_id}, t={self.transform})"

    def debug(self) -> str:
        max_x = max(pos.x for pos in self._body)
        max_y = max(pos.y for pos in self._body)
        grid = np.full((max_x * self._print_scale + self._print_scale, max_y * self._print_scale + self._print_scale), ' ', dtype=str)
        
        for x, y in self._body:
            for i in range(self._print_scale):
                for j in range(self._print_scale):
                    grid[x * self._print_scale + i][y * self._print_scale + j] = '+'
        
        return '\n'.join(''.join(row) for row in grid.tolist())

    @property
    def shape_id(self) -> str:
        return self._shape_id
    
    @property
    def transform(self) -> int:
        return self._transform

    @property
    def body(self) -> List[Vector2]:
        return self._body
    
    @property
    def dimensions(self) -> Vector2:
        return self._dimensions

    @property
    def expanders(self) -> Dict[Vector2, List[Vector2]]:
        return self._expanders

    @property
    def locked(self) -> List[Vector2]:
        return self._locked
    
    @property
    def idx(self) -> int:
        return self._idx
    
    @property
    def size(self) -> int:
        return len(self._body)
    
    def _set_transform(self, transform: int) -> None:
        self._transform = transform

class BlokusPieceTransformations:
    @staticmethod
    def __get_any_piece(shape_id: str) -> BlokusPiece:
        squares = SAMPLE_SHAPE[shape_id]
        dimensions = (0, 0)
        expanders = {}
        locked = set()
        for tile in squares:
            dimensions = (max(dimensions[0], tile[0] + 1), max(dimensions[1], tile[1] + 1))
            for surrounders in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                new_tile = (tile[0] + surrounders[0], tile[1] + surrounders[1])
                if new_tile not in squares:
                    locked.add(new_tile)

        for tile in squares:
            for surrounders in [(1, 1), (-1, -1), (-1, 1), (1, -1)]:
                new_tile = (tile[0] + surrounders[0], tile[1] + surrounders[1])
                if new_tile not in squares and new_tile not in locked:
                    expanders[Vector2(*new_tile)] = Vector2(*surrounders)
                    
        dimensions = Vector2(dimensions[0], dimensions[1])
        # expanders = sorted(list(expanders))
        locked = sorted(list(locked))
        return BlokusPiece(shape_id=shape_id, body=squares, dimensions=dimensions, expanders=expanders, locked=locked)
    
    @staticmethod
    def __rotate_right_shape(piece: BlokusPiece) -> BlokusPiece:
        (h, w) = piece._dimensions
        right = lambda pos: (pos[1], h - pos[0] - 1)
        new_expanders = {right(i): (dy, -dx) for i, (dx, dy) in piece._expanders.items()}
        new_piece = BlokusPiece(piece.shape_id, list(map(right, piece._body)), dimensions=Vector2(w, h), expanders=new_expanders, locked=list(map(right, piece._locked)))
        return new_piece
    
    @staticmethod
    def __reflect_diagonal(piece: BlokusPiece) -> BlokusPiece:
        (h, w) = piece._dimensions
        diagonal = lambda pos: (pos[1], pos[0])
        new_expanders = {diagonal(i): (dy, dx) for i, (dx, dy) in piece._expanders.items()}
        new_piece = BlokusPiece(piece.shape_id, list(map(diagonal, piece._body)), dimensions=Vector2(w, h), expanders=new_expanders, locked=list(map(diagonal, piece._locked)))
        return new_piece
    
    @staticmethod
    def __reflect_horizontal(piece: BlokusPiece) -> BlokusPiece:
        (h, w) = piece._dimensions
        vertical = lambda pos: (pos[0], w - pos[1] - 1)
        new_expanders = {vertical(i): (dx, -dy) for i, (dx, dy) in piece._expanders.items()}
        new_piece = BlokusPiece(piece.shape_id, list(map(vertical, piece._body)), dimensions=Vector2(h, w), expanders=new_expanders, locked=list(map(vertical, piece._locked)))
        return new_piece
    
    @staticmethod
    def __reflect_vertical(piece: BlokusPiece) -> BlokusPiece:
        (h, w) = piece._dimensions
        horizontal = lambda pos: (h - pos[0] - 1, pos[1])
        new_expanders = {horizontal(i): (-dx, dy) for i, (dx, dy) in piece._expanders.items()}
        new_piece = BlokusPiece(piece.shape_id, list(map(horizontal, piece._body)), dimensions=Vector2(h, w), expanders=new_expanders, locked=list(map(horizontal, piece._locked)))
        return new_piece
    
    def __init__(self, id):
        self.id = id
        transformations = [self.__get_any_piece(id)]
        transformations.append(self.__reflect_vertical(transformations[0]))
        transformations.append(self.__reflect_horizontal(transformations[0]))
        transformations.append(self.__reflect_horizontal(transformations[1]))

        transformations.append(self.__reflect_diagonal(transformations[0]))

        transformations.append(self.__reflect_vertical(transformations[4]))
        transformations.append(self.__reflect_horizontal(transformations[4]))
        transformations.append(self.__reflect_horizontal(transformations[5]))
        ## done this way to ensure ##
        # transformation[x] has a vertical reflection of transformation[y] if (x ^ y) & 1
        # transformation[x] has a horizontal reflection of transformation[y] if (x ^ y) & 2

        transformation_shape_list = []
        transformation_index: List[int] = []
        self.indexes: List[int] = []
        self.raw_index: List[int] = [] 
        for i in range(8):
            piece = tuple(sorted(transformations[i]._body)), transformations[i]._dimensions
            if piece not in transformation_shape_list:
                self.indexes.append(len(transformation_shape_list))
                self.raw_index.append(i)
                transformation_shape_list.append(piece)
                transformation_index.append(i)
            else:
                self.indexes.append(transformation_shape_list.index(piece))
        
        self.transformations: List[BlokusPiece] = []
        for i in transformation_index:
            transformations[i]._set_transform(len(self.transformations))
            self.transformations.append(transformations[i])

    def get_transformation(self, transform: int) -> BlokusPiece:
        return self.transformations[transform]
    
    def flip_x(self, transform: int) -> int:
        raw_index = self.raw_index[transform] ^ 1
        new_transform = self.indexes[raw_index]
        return new_transform

    def flip_y(self, transform: int) -> int:
        raw_index = self.raw_index[transform] ^ 2
        new_transform = self.indexes[raw_index]
        return new_transform
    
    def rot180(self, transform: int) -> int:
        raw_index = self.raw_index[transform] ^ 3
        new_transform = self.indexes[raw_index]
        return new_transform
    
class BlokusPieceManager:
    PIECE_VARIANTS_COUNT: int = 91
    pieces: dict[str, BlokusPieceTransformations] = None
    # This will ensure that the pieces are initialized once when the class is first used.
    def __new__(cls, *args, **kwargs):
        if cls.pieces is None:
            # Initialize the pieces only once on first access
            cls.pieces = {shape_id: BlokusPieceTransformations(shape_id) for shape_id in PIECE_SHAPE_IDS}
            assert sum(map(lambda x: len(x.transformations), cls.pieces.values())) == BlokusPieceManager.PIECE_VARIANTS_COUNT
            
            cls.piece_infos = []
            for shape_id, piece in cls.pieces.items():
                for idx, transformation in enumerate(piece.transformations):
                    transformation._idx = len(cls.piece_infos)
                    cls.piece_infos.append((shape_id, idx))

        return super().__new__(cls)
    
    @staticmethod
    def _get_info(piece_id: int) -> Tuple[str, int]:
        return BlokusPieceManager.piece_infos[piece_id]

    @staticmethod
    def get_transformations(shape_id: str) -> List[BlokusPiece]:
        return BlokusPieceManager.pieces[shape_id].transformations
    
    @staticmethod
    def get_piece(*, piece_id: int = None, shape_id: str = None, transform: int = None) -> BlokusPiece:
        if shape_id is not None and transform is not None:
            assert piece_id is None
        elif piece_id is not None:
            assert shape_id is None and transform is None
            shape_id, transform = BlokusPieceManager._get_info(piece_id)
        else:
            raise ValueError("Either shape_id and transform or piece_id must be provided.")
        return BlokusPieceManager.pieces[shape_id].get_transformation(transform)
    
    @staticmethod
    def flip_x(shape_id: str, transform: int) -> int:
        resulting_transform = BlokusPieceManager.pieces[shape_id].flip_x(transform)
        return resulting_transform
    
    @staticmethod
    def flip_y(shape_id: str, transform: int) -> int:
        resulting_transform = BlokusPieceManager.pieces[shape_id].flip_y(transform)
        return resulting_transform
    
    @staticmethod
    def rot180(shape_id: str, transform: int) -> int:
        resulting_transform = BlokusPieceManager.pieces[shape_id].rot180(transform)
        return resulting_transform
    
    # @staticmethod
    # def flip_x(piece: BlokusPiece) -> BlokusPiece:
    #     resulting_transform = BlokusPieceManager.pieces[piece.shape_id].flip_x(piece.transform)
    #     return BlokusPieceManager.get_piece(shape_id=piece.shape_id, transform=resulting_transform)
    
    # @staticmethod
    # def flip_y(piece: BlokusPiece) -> BlokusPiece:
    #     resulting_transform = BlokusPieceManager.pieces[piece.shape_id].flip_y(piece.transform)
    #     return BlokusPieceManager.get_piece(shape_id=piece.shape_id, transform=resulting_transform)
    
    # def reflect_horizontal(self, index):
    #     return self.raw_index[index] ^ 2

    # def reflect_vertical(self, index):
    #     return self.raw_index[index] ^ 1
