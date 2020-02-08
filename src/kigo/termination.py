import abc

from kigo.board import Move
from kigo import scoring
from kigo.types import Agent


class TerminationStrategy(abc.ABC):
    @abc.abstractmethod
    def should_pass(self, game_state):
        ""

    @abc.abstractmethod
    def should_resign(self, game_state):
        ""


class PassWhenOpponentPasses(TerminationStrategy):
    def __init__(self):
        super().__init__()

    def should_pass(self, game_state):
        if game_state.last_move is not None:
            return True if game_state.last_move.is_pass else False

    def should_resign(self, game_state) -> None: pass


class ResignLargeMargin(TerminationStrategy):
    def __init__(self, own_color, margin):
        super().__init__()
        self._own_color = own_color
        self._margin = margin
        self._moves_played = 0

    def should_pass(self, game_state):
        return False

    def should_resign(self, game_state):
        self._moves_played += 1
        if self._moves_played:
            game_result = scoring.compute_game_result(self)
            if game_result.winner != self._own_color and game_result.winning_margin >= self._margin:
                return True
        return False


class TerminationAgent(Agent):
    def __init__(self, agent, strategy=None):
        super().__init__()
        self._agent = agent
        self._strategy = strategy if strategy is not None \
            else TerminationStrategy()

    def select_move(self, game_state):
        if self._strategy.should_pass(game_state):
            return Move.pass_turn()
        elif self._strategy.should_resign(game_state):
            return Move.resign()
        else:
            return self._agent.select_move(game_state)

    def diagnostics(self) -> None: pass


def get(termination):
    if termination == 'opponent_passes':
        return PassWhenOpponentPasses()
    else:
        raise ValueError("Unsupported termination strategy: {}".format(termination))
