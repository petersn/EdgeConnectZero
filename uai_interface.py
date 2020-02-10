#!/usr/bin/python

import sys, string, random, time
#import ataxx_rules
import edgeconnect_rules

def uai_encode_square(xy):
	x, y = xy
	y = 6 - y
	return "%s%i" % (string.ascii_lowercase[x], y + 1)

def uai_encode_move(move):
	return "%s-%s" % (move[0], move[1])
	# ---
	if move == "pass":
		return "0000"
	start, end = move
	if start == "c":
		return uai_encode_square(end)
	return "%s%s" % (uai_encode_square(start), uai_encode_square(end))

def uai_decode_square(s):
	x, y = s.split("-")
	return int(x), int(y)
	# ---
	x, y = string.ascii_lowercase.index(s[0].lower()), int(s[1]) - 1
	y = 6 - y
	return x, y

def uai_decode_move(s):
	q, r = s.split("-")
	return int(q), int(r)
	if s in ("pass", "none", "0000"):
		return "pass"
	elif len(s) == 2:
		return "c", uai_decode_square(s)
	elif len(s) == 4:
		return uai_decode_square(s[:2]), uai_decode_square(s[2:])
	else:
		raise Exception("Bad UAI move string: %r" % s)

def test():
	for s in ["f2", "c3d5"]:
		assert uai_encode_move(uai_decode_move(s)) == s
	for m in [("c", (4, 3)), ((4, 3), (2, 5))]:
		assert uai_decode_move(uai_encode_move(m)) == m
#test()

class FastEngine:
	def __init__(self):
		self.board = edgeconnect_rules.EdgeConnectState.initial()

	def set_state(self, board):
		fast_engine.set_state_from_string(board.to_string().encode("ascii"))
		self.board = board.copy()

	def genmove(self, seconds, use_weighted_exponent="this_argument_is_ignored"):
		start = time.time()
		thinkies_thunked = 0
		visits_to_use = 1000000000 if args.visits is None else args.visits
		visits_multiplier = 2 * {
			"a": 1 - args.first_stone_fraction,
			"b": args.first_stone_fraction,
		}[self.board.move_state[1]]
		# XXX: FIXME: This doesn't *technically* conserve visits, as we round 0.5 up in both cases.
		# To fix this use the opposite rounding mode for "a" as for "b".
		visits_to_use = int(round(visits_multiplier * visits_to_use))
		# Get the current number of visits.
		visits_already_completed = fast_engine.get_visits_in_current_tree()
		print("Already completed:", visits_already_completed, file=sys.stderr)
		visits_to_use = max(1, visits_to_use - visits_already_completed)
		for _ in range(visits_to_use):
			best_move = fast_engine.think()
			thinkies_thunked += 1
			now = time.time()
			if now - start > seconds:
				break
		elapsed = time.time() - start
		print("Steps:", thinkies_thunked, "NPS:", thinkies_thunked / elapsed, "Time:", elapsed, "Intended:", seconds, file=sys.stderr)
		return best_move // 23, best_move % 23

def make_pair():
#	board = ataxx_rules.AtaxxState.initial()
	board = edgeconnect_rules.EdgeConnectState.initial()
	if args.fast:
		return board, FastEngine()
	eng = engine.MCTSEngine()
	if args.visits != None:
#		eng.VISITS = args.visits
		eng.MAX_STEPS = args.visits
	return board, eng

def main(args):
	board, eng = make_pair()
	while True:
		line = input()
		if line == "quit":
			exit()
		elif line == "uai":
			print("id name EdgeConnectZero")
			print("id author Peter Schmidt-Nielsen")
			print("uaiok")
		elif line == "uainewgame":
			board, eng = make_pair()
		elif line == "isready":
			print("readyok")
		elif line.startswith("moves "):
			for move in line[6:].split():
				move = uai_decode_move(move)
				board.make_move(move)
			eng.set_state(board.copy())
		elif line.startswith("position fen "):
			board = edgeconnect_rules.EdgeConnectState.from_string(line[13:])
			eng.set_state(board)
			if args.show_game:
				print("===", file=sys.stderr)
				print(board, file=sys.stderr)
		elif line.startswith("go movetime "):
			ms = int(line[12:])
			ms -= args.safety_ms
			if args.play_randomly:
				move = random.choice(board.legal_moves())
			elif args.visits == None:
				move = eng.genmove(ms * 1e-3, use_weighted_exponent=2.0)
			else:
				# This is safe, because of the visit limit we set above.
				move = eng.genmove(1000000.0, use_weighted_exponent=2.0)
			print("bestmove %s" % (uai_encode_move(move),))
			if args.show_game:
				post_move_board = board.copy()
				post_move_board.make_move(move)
				print(post_move_board, file=sys.stderr)
		elif line == "showboard":
			print(board)
			print("boardok")
		sys.stdout.flush()

if __name__ == "__main__":
#	engine.model.Network.FILTERS = 128
#	engine.model.Network.BLOCK_COUNT = 12

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--network-path", metavar="NETWORK", type=str, help="Name of the model to load.")
	parser.add_argument("--visits", metavar="VISITS", default=None, type=int, help="Number of visits during MCTS.")
	parser.add_argument("--safety-ms", metavar="MS", default=0, type=int, help="Number of milliseconds to shave off of each movetime for safety.")
	parser.add_argument("--show-game", action="store_true", help="Show the game on stderr.")
	parser.add_argument("--play-randomly", action="store_true", help="Actually just play uniformly at random.")
	parser.add_argument("--params", default=None, help="Filters and blocks that the model was saved with.")
	parser.add_argument("--fast", action="store_true", help="Fast engine mode.")
	parser.add_argument("--exploration-parameter", default=None, type=float, help="Fast engine exploration parameter. (cpuct)")
	parser.add_argument("--first-stone-fraction", default=0.5, type=float, help="How to split time between first and second stones of a move. At 0.5 we'll do it evenly, at 1.0 we'll do 2x the visits on the first stone, and zero visits on the second (bad), and at 0.0 the other way around. Set to some value in (0, 1). Only works with --fast.")
	args = parser.parse_args()
	print(args, file=sys.stderr)

	if args.params is not None:
		filters, block_count, use_leaky_fc = map(int, args.params.split(","))
		engine.model.Network.FILTERS = filters
		engine.model.Network.BLOCK_COUNT = block_count
		engine.model.Network.USE_LEAKY_FC = use_leaky_fc

	if args.fast:
		import fast_engine
		fast_engine.initialize()
		if args.exploration_parameter is not None:
			fast_engine.exploration_parameter.value = args.exploration_parameter
	else:
		import engine
		engine.setup_evaluator(use_rpc=False)
		engine.initialize_model(args.network_path)
	main(args)

