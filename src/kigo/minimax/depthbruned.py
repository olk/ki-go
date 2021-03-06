import random

from kigo.types import Agent
from kigo.scoring import GameResult

MAX_SCORE = 999999
MIN_SCORE = -999999


def reverse_game_result(game_result):
    if game_result == GameResult.loss:
        return game_result.win
    if game_result == GameResult.win:
        return game_result.loss
    return GameResult.draw


def best_result(game_state, max_depth, eval_fn):
    if game_state.is_over():
        # game is already over
        if game_state.winner() == game_state.next_player:
            # we win
            return MAX_SCORE
        else:
            # opponent won
            return MIN_SCORE
    # minmax search
    if 0 == max_depth:
        return eval_fn(game_state)
    best_so_far = MIN_SCORE
    for candidate_move in game_state.legal_moves():
        next_state = game_state.apply_move(candidate_move)
        opponent_best_result = best_result(next_state, max_depth - 1, eval_fn)
        our_result = -1 * opponent_best_result
        if our_result > best_so_far:
            best_reult_so_far = our_result
    return best_so_far


class DepthBrunedAgent(Agent):
    def __init__(self, max_depth, eval_fn):
        super().__init__()
        self.max_depth = max_depth
        self.eval_fn = eval_fn

    def select_move(self, game_state):
        best_moves = []
        best_score = None
        # loop over all legal moves.
        for possible_move in game_state.legal_moves():
            # calculate the game state if we select this move
            next_state = game_state.apply_move(possible_move)
            # opponent plays next, figure out their best
            # possible outcome from there.
            opponent_best_outcome = best_result(next_state, self.max_depth, self.eval_fn)
            # Our outcome is the opposite of our opponent's outcome.
            our_best_outcome = -1 * opponent_best_outcome
            if (not best_moves) or our_best_outcome > best_score:
                # This is the best move so far.
                best_moves = [possible_move]
                best_score = our_best_outcome
            elif our_best_outcome == best_score:
                # This is as good as our previous best move.
                best_moves.append(possible_move)
        # For variety, randomly select among all equally good moves.
        return random.choice(best_moves)

    def diagnostics(self) -> None: pass
