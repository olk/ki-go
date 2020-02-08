import mxnet as mx
import re
import subprocess

from io import TextIOWrapper
from pathlib import Path
from subprocess import PIPE, Popen

from kigo.dl.betago import BetaGoAgent
from kigo.termination import PassWhenOpponentPasses, TerminationAgent
from kigo.board import GameState, Move
from kigo.types import Player
from kigo.utility import gtp_position_to_coords, coords_to_gtp_position, print_board
from kigo.sgf.sgf_writer import SGFWriter
from kigo.scoring import compute_game_result


class LocalGtpBot:
    def __init__(self, agent, termination=None, handicap=0, opponent='gnugo', output_sgf="out.sgf",
                 our_color='w'):
        # initialize a bot from an agent and a termination strategy
        self._agent = TerminationAgent(agent, termination)
        self._handicap = handicap
        # _play until the game is stopped by one of the _players
        self._stopped = False
        self._game_state = GameState.new_game(19)
        # at the end we write the the game to the provided file in SGF forma
        self._sgf = SGFWriter(output_sgf)
        self._our_color = Player.black if our_color == 'b' else Player.white
        self._their_color = self._our_color.other
        # opponent will either be GNU Go or Pachi
        cmd = self.opponent_cmd(opponent)
        pipe = subprocess.PIPE
        # read and write GTP commands from the command line
        self._proc = Popen(cmd, stdin=PIPE, stdout=PIPE)
        self._stdin = TextIOWrapper(self._proc.stdin, encoding='utf-8', line_buffering=True)
        self._stdout = TextIOWrapper(self._proc.stdout, encoding='utf-8')

    @staticmethod
    def opponent_cmd(opponent):
        if opponent == 'gnugo':
            return ["gnugo", "--mode", "gtp"]
        elif opponent == 'pachi':
            return ["pachi"]
        else:
            raise ValueError("Unknown bot name {}".format(opponent))

    def _send_command(self, cmd):
        self._stdin.write(cmd)

    def _get_response(self):
        succeeded = False
        result = ''
        while not succeeded:
            line = self._stdout.readline()
            if line[0] == '=':
                succeeded = True
                line = line.strip()
                result = re.sub('^= ?', '', line)
        return result

    def _command_and_response(self, cmd):
        self._send_command(cmd)
        resp = self._get_response()
        return resp

    def _set_handicap(self):
        if self._handicap == 0:
            self._command_and_response("komi 7.5\n")
            self._sgf.append("KM[7.5]\n")
        else:
            stones = self._command_and_response("fixed_handicap {}\n".format(self._handicap))
            sgf_handicap = "HA[{}]AB".format(self._handicap)
            for pos in stones.split(" "):
                move = Move(gtp_position_to_coords(pos))
                self._game_state = self._game_state.apply_move(move)
                sgf_handicap = sgf_handicap + "[" + self._sgf.coordinates(move) + "]"
            self._sgf.append(sgf_handicap + "\n")

    def _play(self):
        while not self._stopped:
            if self._game_state.next_player == self._our_color:
                self._play_our_move()
            else:
                self._play_their_move()
            print(chr(27) + "[2J")
            print_board(self._game_state.board)
            print("Estimated result: ")
            print(compute_game_result(self._game_state))

    def _play_our_move(self):
        move = self._agent.select_move(self._game_state)
        self._game_state = self._game_state.apply_move(move)

        our_name = self._our_color.name
        our_letter = our_name[0].upper()
        sgf_move = ""
        if move.is_pass:
            self._command_and_response("play {} pass\n".format(our_name))
        elif move.is_resign:
            self._command_and_response("play {} resign\n".format(our_name))
        else:
            pos = coords_to_gtp_position(move)
            self._command_and_response("play {} {}\n".format(our_name, pos))
            sgf_move = self._sgf.coordinates(move)
        self._sgf.append(";{}[{}]\n".format(our_letter, sgf_move))

    def _play_their_move(self):
        their_name = self._their_color.name
        their_letter = their_name[0].upper()

        pos = self._command_and_response("genmove {}\n".format(their_name))
        if pos.lower() == 'resign':
            self._game_state = self._game_state.apply_move(Move.resign())
            self._stopped = True
        elif pos.lower() == 'pass':
            self._game_state = self._game_state.apply_move(Move.pass_turn())
            self._sgf.append(";{}[]\n".format(their_letter))
            if self._game_state.last_move.is_pass:
                self._stopped = True
        else:
            move = Move(gtp_position_to_coords(pos))
            self._game_state = self._game_state.apply_move(move)
            self._sgf.append(";{}[{}]\n".format(their_letter, self._sgf.coordinates(move)))

    def run(self):
        self._command_and_response("boardsize 19\n")
        self._set_handicap()
        self._play()
        self._sgf.write_sgf()


if __name__ == "__main__":
    ctx = mx.gpu()
    agent = BetaGoAgent.create(Path('./checkpoints/').resolve().joinpath('betago.params'), ctx)
    bot = LocalGtpBot(agent=agent, termination=PassWhenOpponentPasses(), handicap=0, opponent='gnugo')
    bot.run()
