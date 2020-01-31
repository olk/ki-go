import mxnet as mx

from pathlib import Path
from six.moves import input

from kigo.dl.betago import BetaGoAgent
from kigo.board import GameState, Move
from kigo.types import Player
from kigo.utility import print_board, print_move, point_from_coords


def main():
    game = GameState.new_game(19)
    checkpoint_p = Path('./checkpoints/').resolve().joinpath('betago.params')
    ctx = mx.gpu()
    agent = BetaGoAgent.create(checkpoint_p, ctx)

    #for i in range(3):
    #    human_move = 'A%d' % (i+1)
    #    print(human_move)
    #    point = point_from_coords(human_move.strip())
    #    print(point)
    #    move = Move.play(point)
    #    game = game.apply_move(move)
    #    move = agent.select_move(game)
    #    game = game.apply_move(move)
    #    print_board(game.board)

    while not game.is_over():
        # before each move, clear screen
        print(chr(27) + "[2J")  # <2>
        print_board(game.board)
        if Player.black == game.next_player:
            human_move = input('-- ')
            point = point_from_coords(human_move.strip())
            move = Move.play(point)
        else:
            move = agent.select_move(game)
        game = game.apply_move(move)
    print_board(game.board)
    winner = game.winner()
    if winner is None:
        print("It's a draw.")
    else:
        print('Winner: ' + str(winner))


if __name__ == '__main__':
    main()
