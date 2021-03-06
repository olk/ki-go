import numpy as np

from kigo.encoders.base import Encoder
from kigo.board import Move, Point


class SevenPlaneEncoder(Encoder):
    def __init__(self, board_size):
        super().__init__()
        self.board_width, self.board_height = board_size
        self.num_planes = 7

    def name(self):
        return 'sevenplane'

    def encode(self, game_state):
        # fill a matrix with 1 if the point contains one of the current player's stones,
        # -1 if the point contains the opponent's stones and 0 if the point is empty
        board_tensor = np.zeros(self.shape(), dtype='int')
        base_plane = {game_state.next_player: 0,
                      game_state.next_player.other: 3}
        for row in range(self.board_height):
            for col in range(self.board_width):
                p = Point(row=row + 1, col=col + 1)
                go_string = game_state.board.get_string(p)
                if go_string is None:
                    if game_state.does_move_violate_ko(game_state.next_player,
                                                       Move.play(p)):
                        # encode KO 
                        board_tensor[6][row][col] = 1
                else:
                    liberty_plane = min(3, go_string.num_liberties) - 1
                    liberty_plane += base_plane[go_string.color]
                    # encode based on liberties 
                    board_tensor[liberty_plane][row][col] = 1
        return board_tensor

    def encode_point(self, point):
        return int(self.board_width * (point.row - 1) + (point.col - 1))

    def decode_point_index(self, index):
        row = index // self.board_width
        col = index % self.board_width
        return Point(row=row + 1, col=col + 1)

    def num_points(self):
        return self.board_width * self.board_height

    def shape(self):
        return self.num_planes, self.board_height, self.board_width

    @classmethod
    def create(board_size):
        return SevenPlaneEncoder(board_size)
