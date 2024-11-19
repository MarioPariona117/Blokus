from typing import List, Tuple

PIECE_IDS = ['1', '2', 'I3', 'V3', 'I4', 'L4', 'O', 'T4', 'Z4', 'F', 'I5', 'L5', 'N', 'P', 'T5', 'U', 'V5', 'W', 'X', 'Y', 'Z5']

class BlokusPiece:
    def __init__(self, shape: List[Tuple[int, int]], size: Tuple[int, int], idx=None):
        self.shape = shape
        self.size = size
        
    def print(self):
        N = 3
        max_x = max(pos[0] for pos in self.shape)
        max_y = max(pos[1] for pos in self.shape)
        grid = [[' ' for _ in range(max_y * N + N)] for _ in range(max_x * N + N)]
        
        for x, y in self.shape:
            for i in range(N):
                for j in range(N):
                    grid[x * N + i][y * N + j] = '+'
        
        for row in grid:
            print(''.join(row))
        print()

class BlokusPieceTransformations:
    @staticmethod
    def __get_any_piece(id):
        match id:
            case '1':
                shape = [(0, 0)]
            case '2':
                shape = [(0, 0), (0, 1)]
            # 3-block pieces (2)
            case 'I3':
                shape = [(0, 0), (0, 1), (0, 2)]
            case 'V3':
                shape = [(0, 0), (0, 1), (1, 1)]
            ## 4-block pieces (5)
            case 'I4':
                shape = [(0, 0), (0, 1), (0, 2), (0, 3)]
            case 'L4':
                shape = [(0, 0), (1, 0), (1, 1), (1, 2)]
            case 'O':
                shape = [(0, 0), (0, 1), (1, 0), (1, 1)]
            case 'T4':
                shape = [(0, 0), (0, 1), (0, 2), (1, 1)]
            case 'Z4':
                shape = [(0, 0), (0, 1), (1, 1), (1, 2)]
            # 5-block pieces (12)
            case 'F':
                shape = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 1)]
            case 'I5':
                shape = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
            case 'L5':
                shape = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)]
            case 'N':
                shape = [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3)]
            case 'P':
                shape = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]
            case 'T5':
                shape = [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)]
            case 'U':
                shape = [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)]
            case 'V5':
                shape = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
            case 'W':
                shape = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)]
            case 'X':
                shape = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]
            case 'Y':
                shape = [(0, 0), (1, 0), (2, 0), (3, 0), (2, 1)]
            case 'Z5':
                shape = [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)]
        size = (0, 0)
        for tile in shape:
            size = (max(size[0], tile[0] + 1), max(size[1], tile[1] + 1))
        return BlokusPiece(shape, size)
    
    @staticmethod
    def __rotate_right_shape(piece):
        (h, w) = piece.size
        new_piece = BlokusPiece([(j, h - i - 1) for (i, j) in piece.shape], size=(w, h))
        return new_piece
    
    @staticmethod
    def __reflect_diagonal(piece):
        (h, w) = piece.size
        new_piece = BlokusPiece(shape=[(j, i) for (i, j) in piece.shape], size=(w, h))
        return new_piece
    
    @staticmethod
    def __reflect_vertical(piece):
        (h, w) = piece.size
        new_piece = BlokusPiece(shape=[(i, w - j - 1) for (i, j) in piece.shape], size=(h, w))
        return new_piece
    
    @staticmethod
    def __reflect_horizontal(piece):
        (h, w) = piece.size
        new_piece = BlokusPiece(shape=[(h - i - 1, j) for (i, j) in piece.shape], size=(h, w))
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

        self.transformation_shape_list = []
        self.indexes = []
        self.aux = []
        for i in range(8):
            piece = tuple(sorted(transformations[i].shape)), transformations[i].size
            if piece not in self.transformation_shape_list:
                self.indexes.append(len(self.transformation_shape_list))
                self.aux.append(i)
                self.transformation_shape_list.append(piece)
            else:
                self.indexes.append(self.transformation_shape_list.index(piece))
        
        self.transformations = [BlokusPiece(list(shape), size) for shape, size in self.transformation_shape_list]

