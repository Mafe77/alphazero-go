import numpy as np

from dlgo.encoders.base import Encoder
from dlgo.goboard import Move
from dlgo.gotypes import Player, Point

'''
    Planes:
        0 - 3: Black stones with 1, 2, 3, 4+ liberties
        4 - 7: White stones with 1, 2, 3, 4+ liberties
        8: Black plays next
        9: White plays next
        10: Illegal moves due to ko
'''

class SimpleEncoder(Encoder):
    def __init__(self, board_size):
        self.board_width, self.board_heigth = board_size
        self.num_planes = 11
    
    def name(self):
        return 'simple'
    
    def encode(self, game_state):
        board_tensor = np.zeros(self.shape())
        # Planes 8 and 9
        if game_state.next_player == Player.black:
            board_tensor[8] = 1
        else:
            board_tensor[9] = 1
        
        for r in range(self.board_heigth):
            for c in range(self.board_width):
                p = Point(row=r+1, col=c+1)
                go_string = game_state.board.get_go_string(p)

                if go_string is None:
                    # Plane 10
                    if game_state.does_move_violate_ko(game_state.next_player, Move.play(p)):
                        board_tensor[10][r][c] = 1
                else:
                    # Planes 0-7
                    liberty_plane = min(4, go_string.num_liberties) - 1
                    if go_string.color == Player.white:
                        liberty_plane += 4
                    board_tensor[liberty_plane][r][c]

        return board_tensor
    
    def encode_point(self, point):
        """Turn a board point into an integer index."""
        # Points are 1-indexed
        return self.board_width * (point.row - 1) + (point.col - 1)

    def decode_point_index(self, index):
        """Turn an integer index into a board point."""
        row = index // self.board_width
        col = index % self.board_width
        return Point(row=row + 1, col=col + 1)

    def num_points(self):
        return self.board_width * self.board_height

    def shape(self):
        return self.num_planes, self.board_height, self.board_width


def create(board_size):
    return SimpleEncoder(board_size)