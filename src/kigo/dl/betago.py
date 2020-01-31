import mxnet as mx
import numpy as np

from kigo.board import Move
from kigo.encoders.sevenplane import SevenPlaneEncoder
from kigo.helpers import is_point_an_eye
from kigo.networks import betago
from kigo.types import Agent


class BetaGoAgent(Agent):
    def __init__(self, model, encoder, ctx):
        super().__init__()
        self._model = model
        self._encoder = encoder
        self._ctx = ctx

    def predict(self, game_state):
        encoded_state = self._encoder.encode(game_state)
        input_tensor = mx.nd.array(encoded_state).expand_dims(axis=0).as_in_context(self._ctx)
        return self._model.predict(input_tensor)

    def select_move(self, game_state):
        num_moves = self._encoder.board_width * self._encoder.board_height
        move_probs = self.predict(game_state)
        move_probs = move_probs[0].asnumpy()
        # increase the distance between the move likely and least likely moves
        move_probs = move_probs ** 3
        # prevent move probs from getting stuck at 0 or 1
        eps = 1e-6
        move_probs = np.clip(move_probs, eps, 1 - eps)
        # re-normalize to get another probability distribution
        move_probs = move_probs / np.sum(move_probs)
        # turn the probabilities into a ranked list of moves.
        candidates = np.arange(num_moves)
        # sample potential candidates
        ranked_moves = np.random.choice(
            candidates, num_moves, replace=False, p=move_probs)
        for point_idx in ranked_moves:
            point = self._encoder.decode_point_index(point_idx)
            # starting from the top, find a valid move that doesn't reduce eye-space
            if game_state.is_valid_move(Move.play(point)) and \
                    not is_point_an_eye(game_state.board, point, game_state.next_player):
                return Move.play(point)
        # if no legal and non-self-destructive moves are left, pass
        return Move.pass_turn()

    def diagnostics(self) -> None: pass

    def serialize(self, checkpoint_p):
        self._model.save(checkpoint_p)

        encoder = SevenPlaneEncoder((19, 19))
        model = betago.Model.create(encoder, checkpoint_p, ctx)
        return BetaGoAgent(model, encoder)

    @classmethod
    def create(cls, checkpoint_p, ctx):
        encoder = SevenPlaneEncoder((19, 19))
        model = betago.Model.create(encoder, checkpoint_p, ctx)
        return BetaGoAgent(model, encoder, ctx)
