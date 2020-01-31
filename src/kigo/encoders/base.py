import abc
import importlib


class Encoder(abc.ABC):
    @abc.abstractmethod
    def name(self):
        "lets us support logging or saving the name of the encoder our model is using"

    @abc.abstractmethod
    def encode(self, game_state):
        "turn a Go board into a numeric data"

    @abc.abstractmethod
    def encode_point(self, point):
        "turn a Go board point into an integer index"

    @abc.abstractmethod
    def decode_point_index(self, index):
        "turn an integer index back into a Go board point"

    @abc.abstractmethod
    def num_points(self):
        "number of points on the board, i.e. board width times board height"

    @abc.abstractmethod
    def shape(self):
        "shape of the encoded board structure"


def get_encoder_by_name(name, board_size):  # <1>
    if isinstance(board_size, int):
        board_size = (board_size, board_size)  # <2>
    module = importlib.import_module('kigo.encoders.' + name)
    constructor = getattr(module, 'create')  # <3>
    return constructor(board_size)
