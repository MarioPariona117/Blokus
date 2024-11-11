import unittest

from gymnasium_env.envs.blokus_piece import BlokusPieceTransformations

# BlokusPieceTransformations
class TestBlokusPieceTransformations(unittest.TestCase):
    pieces_ids = ['1', '2', 'I3', 'V3', 'I4', 'L4', 'O', 'T4', 'Z4', 'F', 'I5', 'L5', 'N', 'P', 'T5', 'U', 'V5', 'W', 'X', 'Y', 'Z5']
    def setUp(self):
        self.pieces = [BlokusPieceTransformations(id=id) for id in self.pieces_ids]
    
    def test_initial_shape(self):
        # n = 0
        # for piece_transform in self.pieces:
        #     n += len(piece_transform.transformations)
        # print(n)
        # for piece in self.pieces:
            # print(piece.id, len(piece.transformations))
            # for transformation in piece.transformations:
            #     transformation.print()
                # print(j.shape)
        # self.assertEqual(self.piece.shape, [[1, 1], [1, 1]])
        pass
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