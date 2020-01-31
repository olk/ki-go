import time

from kigo.ts.mcts import MCTSAgent
from kigo.board import GameState
from kigo.types import Player
from kigo.utility import print_board, print_move


def main():
    board_size = 5
    game = GameState.new_game(board_size)
    bots = {
            Player.black: MCTSAgent(num_rounds=500, temperature=1.4),
            Player.white: MCTSAgent(num_rounds=500, temperature=1.4),
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
