import time

from kigo.minimax.alphabeta import AlphaBetaAgent
from kigo.board import GameState
from kigo.types import Player, Point
from kigo.utility import print_board, print_move


def capture_diff(game_state):
    black_stones = 0
    white_stones = 0
    for r in range(1, game_state.board.num_rows + 1):
        for c in range(1, game_state.board.num_cols + 1):
            p = Point(r, c)
            color = game_state.board.get(p)
            if color == Player.black:
                black_stones += 1
            elif color == Player.white:
                white_stones += 1
    diff = black_stones - white_stones
    if game_state.next_player == Player.black:
        return diff
    return -1 * diff


def main():
    board_size = 3
    game = GameState.new_game(board_size)
    bots = {
            Player.black: AlphaBetaAgent(3, capture_diff),
            Player.white: AlphaBetaAgent(3, capture_diff),
    }
    while not game.is_over():
        # sleep 300ms so that bot moves are not printed too fast
        time.sleep(0.3)
        # before each move, clear screen
        print(chr(27) + "[2J")  # <2>
        bot_move = bots[game.next_player].select_move(game)
        print_board(game.board)
        print_move(game.next_player, bot_move)
        game = game.apply_move(bot_move)
    print_board(game.board)
    winner = game.winner()
    if winner is None:
        print("It's a draw.")
    else:
        print('Winner: ' + str(winner))


if __name__ == '__main__':
    main()
