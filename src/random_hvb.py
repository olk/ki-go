import time

from six.moves import input

from kigo.naive.random import RandomAgent
from kigo.board import GameState, Move
from kigo.types import Player
from kigo.utility import print_board, print_move, point_from_coords


def main():
    board_size = 9
    game = GameState.new_game(board_size)
    bot = RandomAgent()
    while not game.is_over():
        # before each move, clear screen
        print(chr(27) + "[2J")  # <2>
        print_board(game.board)
        if Player.black == game.next_player:
            human_move = input('-- ')
            point = point_from_coords(human_move.strip())
            move = Move.play(point)
        else:
            move = bot.select_move(game)
        game = game.apply_move(move)

    print_board(game.board)
    winner = game.winner()
    if winner is None:
        print("It's a draw.")
    else:
        print('Winner: ' + str(winner))


if __name__ == '__main__':
    main()
