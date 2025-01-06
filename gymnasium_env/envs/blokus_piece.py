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
            shape: List[Vector2], 
            dimensions: Vector2, 
            expanders: Dict[Vector2, List[Vector2]], 
            locked: List[Vector2],
            print_scale: int = 2
        ):
        self.idx = None
        self.shape_id = shape_id
        self.shape = shape
        self.dimensions = dimensions
        self.expanders = expanders
        self.locked = locked
        self.print_scale = print_scale

    def __str__(self) -> str:
        max_x = max(pos.x for pos in self.shape)
        max_y = max(pos.y for pos in self.shape)
        grid = np.full((max_x * self.print_scale + self.print_scale, max_y * self.print_scale + self.print_scale), ' ', dtype=str)
        
        for x, y in self.shape:
            for i in range(self.print_scale):
                for j in range(self.print_scale):
                    grid[x * self.print_scale + i][y * self.print_scale + j] = '+'
        
        return '\n'.join(''.join(row) for row in grid.tolist())

    def get_shape(self) -> List[Vector2]:
        return self.shape
    
    def get_dimensions(self) -> Vector2:
        return self.dimensions

    def get_expanders(self) -> Dict[Vector2, List[Vector2]]:
        return self.expanders
    
class BlokusPieceTransformations:
    @staticmethod
    def __get_any_piece(shape_id: str) -> BlokusPiece:
        shape = SAMPLE_SHAPE[shape_id]
        dimensions = (0, 0)
        expanders = {}
        locked = set()
        for tile in shape:
            dimensions = (max(dimensions[0], tile[0] + 1), max(dimensions[1], tile[1] + 1))
            for surrounders in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                new_tile = (tile[0] + surrounders[0], tile[1] + surrounders[1])
                if new_tile not in shape:
                    locked.add(new_tile)

        for tile in shape:
            for surrounders in [(1, 1), (-1, -1), (-1, 1), (1, -1)]:
                new_tile = (tile[0] + surrounders[0], tile[1] + surrounders[1])
                if new_tile not in shape and new_tile not in locked:
                    expanders[Vector2(*new_tile)] = Vector2(*surrounders)
                    
        dimensions = Vector2(dimensions[0], dimensions[1])
        # expanders = sorted(list(expanders))
        locked = sorted(list(locked))
        return BlokusPiece(shape_id=shape_id, shape=shape, dimensions=dimensions, expanders=expanders, locked=locked)
    
    @staticmethod
    def __rotate_right_shape(piece: BlokusPiece) -> BlokusPiece:
        (h, w) = piece.dimensions
        right = lambda pos: (pos[1], h - pos[0] - 1)
        new_expanders = {right(i): (dy, -dx) for i, (dx, dy) in piece.expanders.items()}
        new_piece = BlokusPiece(piece.shape_id, list(map(right, piece.shape)), dimensions=Vector2(w, h), expanders=new_expanders, locked=list(map(right, piece.locked)))
        return new_piece
    
    @staticmethod
    def __reflect_diagonal(piece: BlokusPiece) -> BlokusPiece:
        (h, w) = piece.dimensions
        diagonal = lambda pos: (pos[1], pos[0])
        new_expanders = {diagonal(i): (dy, dx) for i, (dx, dy) in piece.expanders.items()}
        new_piece = BlokusPiece(piece.shape_id, list(map(diagonal, piece.shape)), dimensions=Vector2(w, h), expanders=new_expanders, locked=list(map(diagonal, piece.locked)))
        return new_piece
    
    @staticmethod
    def __reflect_vertical(piece: BlokusPiece) -> BlokusPiece:
        (h, w) = piece.dimensions
        vertical = lambda pos: (pos[0], w - pos[1] - 1)
        new_expanders = {vertical(i): (dx, -dy) for i, (dx, dy) in piece.expanders.items()}
        new_piece = BlokusPiece(piece.shape_id, list(map(vertical, piece.shape)), dimensions=Vector2(h, w), expanders=new_expanders, locked=list(map(vertical, piece.locked)))
        return new_piece
    
    @staticmethod
    def __reflect_horizontal(piece: BlokusPiece) -> BlokusPiece:
        (h, w) = piece.dimensions
        horizontal = lambda pos: (h - pos[0] - 1, pos[1])
        new_expanders = {horizontal(i): (-dx, dy) for i, (dx, dy) in piece.expanders.items()}
        new_piece = BlokusPiece(piece.shape_id, list(map(horizontal, piece.shape)), dimensions=Vector2(h, w), expanders=new_expanders, locked=list(map(horizontal, piece.locked)))
        return new_piece
    
    def __init__(self, id):
        self.id = id
        transformations = [self.__get_any_piece(id)]
        transformations.append(self.__reflect_horizontal(transformations[0]))
        transformations.append(self.__reflect_vertical(transformations[0]))
        transformations.append(self.__reflect_vertical(transformations[1]))

        transformations.append(self.__reflect_diagonal(transformations[0]))

        transformations.append(self.__reflect_horizontal(transformations[4]))
        transformations.append(self.__reflect_vertical(transformations[4]))
        transformations.append(self.__reflect_vertical(transformations[5]))
        ## done this way to ensure ##
        # transformation[x] has a vertical reflection of transformation[y] if (x ^ y) & 1
        # transformation[x] has a horizontal reflection of transformation[y] if (x ^ y) & 2

        transformation_shape_list = []
        transformation_index: List[int] = []
        self.indexes = []
        self.raw_index: List[int] = [] 
        for i in range(8):
            piece = tuple(sorted(transformations[i].shape)), transformations[i].dimensions
            if piece not in transformation_shape_list:
                self.indexes.append(len(transformation_shape_list))
                self.raw_index.append(i)
                transformation_shape_list.append(piece)
                transformation_index.append(i)
            else:
                self.indexes.append(transformation_shape_list.index(piece))
        
        self.transformations = [transformations[i] for i in transformation_index]

    def get_transformation(self, transform: int) -> BlokusPiece:
        return self.transformations[transform]
    
    def get_horizontal_reflected_transformation(self, transform: int) -> int:
        raw_index = self.raw_index[transform] ^ 2
        new_transform = self.indexes[raw_index]
        return new_transform

    # def reflect_vertical(self, index):
    #     return self.raw_index[index] ^ 1
    
    # def reflect_diagonal(self, index):
    #     return self.raw_index[index] ^ 4

class BlokusPieceManager:
    PIECE_VARIANTS_COUNT = 91
    pieces = None  # Class-level variable to store the pieces
    
    # This will ensure that the pieces are initialized once when the class is first used.
    def __new__(cls, *args, **kwargs):
        if cls.pieces is None:
            # Initialize the pieces only once on first access
            cls.pieces = {shape_id: BlokusPieceTransformations(shape_id) for shape_id in PIECE_SHAPE_IDS}
            assert sum(map(lambda x: len(x.transformations), cls.pieces.values())) == BlokusPieceManager.PIECE_VARIANTS_COUNT
            
            cls.piece_infos = []
            for shape_id, piece in cls.pieces.items():
                for idx, transformation in enumerate(piece.transformations):
                    transformation.idx = len(cls.piece_infos)
                    cls.piece_infos.append((shape_id, idx))

        return super().__new__(cls)
    
    @staticmethod
    def get_transformations(shape_id: str) -> List[BlokusPiece]:
        return BlokusPieceManager.pieces[shape_id].transformations

    @staticmethod
    def get_info(piece_id: int) -> Tuple[str, int]:
        return BlokusPieceManager.piece_infos[piece_id]

    @staticmethod
    def get_piece(shape_id: str, transform: int) -> BlokusPiece:
        return BlokusPieceManager.pieces[shape_id].get_transformation(transform)

    @staticmethod
    def get_piece_shape(shape_id: str, transform: int) -> List[Vector2]:
        return BlokusPieceManager.pieces[shape_id].get_transformation(transform).get_shape()

    def get_piece_size(shape_id: str) -> int:
        return len(BlokusPieceManager.pieces[shape_id].get_transformation(0).shape)
    
    @staticmethod
    def get_piece_dimensions(shape_id: str, transform: int) -> Vector2:
        return BlokusPieceManager.pieces[shape_id].get_transformation(transform).get_dimensions()

    @staticmethod
    def get_piece_expanders(shape_id: str, transform: int) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        return BlokusPieceManager.pieces[shape_id].get_transformation(transform).get_expanders()
    
    @staticmethod
    def get_piece_idx(shape_id: str, transform: int) -> int:
        return BlokusPieceManager.pieces[shape_id].get_transformation(transform).idx
    
    @staticmethod
    def reflect_horizontal(shape_id: str, transform: int) -> BlokusPiece:
        rt = BlokusPiece.raw_index(shape_id, transform)
        rotated_transform = BlokusPieceManager.pieces[shape_id].get_horizontal_reflected_transformation()
        return BlokusPieceManager.get_piece(shape_id, rotated_transform)

    # def reflect_horizontal(self, index):
    #     return self.raw_index[index] ^ 2

    # def reflect_vertical(self, index):
    #     return self.raw_index[index] ^ 1
