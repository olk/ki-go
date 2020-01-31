import random

from kigo.board import Move
from kigo.helpers import is_point_an_eye
from kigo.types import Agent, Point


class RandomAgent(Agent):
    def __init__(self):
        super().__init__()

    def select_move(self, game_state):
        # choose a random valid move that preserves our own eyes
        candidates = []
        for r in range(1, game_state.board.num_rows + 1):
            for c in range(1, game_state.board.num_cols + 1):
                candidate = Point(row=r, col=c)
                if game_state.is_valid_move(Move.play(candidate)) and\
                        not is_point_an_eye(game_state.board,
                                            candidate,
                                            game_state.next_player):
                    candidates.append(candidate)
        if not candidates:
            return Move.pass_turn()
        return Move.play(random.choice(candidates))

    def diagnostics(self) -> None: pass
