import unittest
from src.environment.blokus_piece import  BlokusPieceGroup

class TestBlokusPieceGroup(unittest.TestCase):
    pieces_ids = ['1', '2', 'I3', 'V3', 'I4', 'L4', 'O', 'T4', 'Z4', 'F', 'I5', 'L5', 'N', 'P', 'T5', 'U', 'V5', 'W', 'X', 'Y', 'Z5']
    def setUp(self):
        self.pieces = [BlokusPieceGroup(id=id) for id in self.pieces_ids]

    def test_initial_shape(self):
        for piece in self.pieces:
            print(piece.id, len(piece.pieces))
            for piece_state in piece.pieces:
                piece_state.print()
                # print(j.shape)
        # self.assertEqual(self.piece.shape, [[1, 1], [1, 1]])

    # def test_rotate(self):
    #     pass
    #     # self.piece.rotate()
    #     # self.assertEqual(self.piece.shape, [[1, 1], [1, 1]])  # Update with expected rotated shape

    # def test_flip(self):
    #     pass
    #     # self.piece.flip()
    #     # self.assertEqual(self.piece.shape, [[1, 1], [1, 1]])  # Update with expected flipped shape

if __name__ == '__main__':
    unittest.main()