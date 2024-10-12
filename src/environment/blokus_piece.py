class BlokusPiece:
    def __init__(self, shape, size):
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

class BlokusPieceGroup:
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
                shape = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]
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
                shape = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)]
        size = (0, 0)
        for tile in shape:
            size = (max(size[0], tile[0] + 1), max(size[1], tile[1] + 1))
        return BlokusPiece(shape, size)
    
    @staticmethod
    def __rotate_right_shape(piece):
        (h, w) = piece.size
        new_piece = BlokusPiece([(j, h - i - 1) for (i, j) in piece.shape], (w, h))
        return new_piece
    
    @staticmethod
    def __reflect_shape(piece):
        (h, w) = piece.size
        new_piece = BlokusPiece(shape=[(j, i) for (i, j) in piece.shape], size=(w, h))
        return new_piece
    
    def __init__(self, id):
        self.id = id
        self.pieces = [self.__get_any_piece(id)]
        for i in range(3):
            self.pieces.append(self.__rotate_right_shape(self.pieces[i]))

        for i in range(4):
            self.pieces.append(self.__reflect_shape(self.pieces[i]))

        self.piece_list = list(set([(tuple(sorted(piece.shape, key=lambda x: (x[0], x[1]))), piece.size) for piece in self.pieces]))
        self.pieces = [BlokusPiece(list(shape), size) for shape, size in self.piece_list]