#!/usr/bin/python

import os, time, random, json, argparse, pprint
#import ataxx_rules
import edgeconnect_rules
import engine
import uai_ringmaster

MAX_STEP_RATIO     = 10
MAXIMUM_GAME_PLIES = 400
LOGIT_TEMPERATURE  = 0.0
OPENING_RANDOMIZATION_SCHEDULE = [
	0.2 * (0.5 ** (i/2))
	for i in range(10)
]

def generate_game(args):
	board = edgeconnect_rules.EdgeConnectState.initial()
	if not args.random_play:
		m = engine.MCTS(board.copy(), use_dirichlet_noise=True)
	entry = {"boards": [], "moves": []}
	all_steps = 0
	for ply in range(MAXIMUM_GAME_PLIES):
		if args.random_play:
			training_move = selected_move = random.choice(board.legal_moves())
		else:
			if args.supervised:
				args.uai_player.set_state(board)
				try:
					training_move = selected_move = args.uai_player.genmove(args.supervised_ms)
				except ValueError as e:
					print("Board state:")
					print(board.fen())
					print(board)
					raise e
			else:
				# Do steps until the root is sufficiently visited.
				most_visits = 0 
				while most_visits < args.visit_count and m.root_node.all_edge_visits < args.visit_count * MAX_STEP_RATIO:
					all_steps += 1
					edge = m.step()
					most_visits = max(most_visits, edge.edge_visits)
				# Pick a move with noise.
				training_move = selected_move = engine.sample_with_exponential_weight(1.0)

			randomization_probability = 0.0
			if ply < len(OPENING_RANDOMIZATION_SCHEDULE):
				randomization_probability = OPENING_RANDOMIZATION_SCHEDULE[ply]
			if random.random() < randomization_probability:
				selected_move = random.choice(board.legal_moves())
		entry["boards"].append(board.to_string())
		entry["moves"].append(training_move)
		# Execute the move.
		if not args.random_play:
			m.play(board.to_move, selected_move)
		board.make_move(selected_move)
		if board.result() != None:
			break
		if args.show_game:
			print(board)
#			engine.global_evaluator.populate(m.root_node.board)
#			m.root_node.board.evaluations.populate_noisy_posterior()
#			__import__("pprint").pprint(sorted(m.root_node.board.evaluations.posterior.items(), key=lambda x: x[1]))
#			__import__("pprint").pprint(sorted(m.root_node.board.evaluations.noisy_posterior.items(), key=lambda x: x[1]))
			input(">")
		if args.die_if_present and os.path.exists(args.die_if_present):
			print("Exiting due to signal file!")
			exit()
	entry["result"] = board.result();
	print("[%3i] Generated a %i ply game (%.2f avg steps) with result %r." % (
		args.group_index,
		len(entry["boards"]),
		all_steps / float(ply + 1),
		entry["result"],
	))
	return entry

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(
		description="""
			Tool for generating games in a .json format suitable for feeding into train.py.
			This tool is very reconfigurable, and can be used to generate uniformly random play, or record self-play games generated by a UAI engine into the .json format.
			However, this tool (either in RPC mode or regular) will use the Python MCTS implementation to generate the games, and therefore isn't the fastest.
			If you want to generate self-play MCTS games as quickly as possible then use accelerated_generate_games.py instead, which is much more limited in function, but much faster at what it does.
		""",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument("--network", metavar="PATH", default="", help="Path of the model to load.")
	parser.add_argument("--output-games", metavar="PATH", type=str, default=None, help="Path to write .json games to. Writes in append mode, so it won't overwrite existing games.")
	parser.add_argument("--group-index", metavar="N", default=0, type=int, help="Our index in the work group.")
	parser.add_argument("--use-rpc", action="store_true", help="Use RPC for NN evaluation.")
	parser.add_argument("--random-play", action="store_true", help="Generate games by totally random play.")
	parser.add_argument("--visit-count", metavar="N", default=200, type=int, help="When generating moves for games perform MCTS steps until the PV move has at least N visits.")
	parser.add_argument("--die-if-present", metavar="PATH", default=None, type=str, help="Die once a file is present at the target path.")
	parser.add_argument("--show-game", action="store_true", help="Show the game while it's generating.")
	parser.add_argument("--game-count", metavar="N", default=None, type=int, help="Maximum number of games to generate.")
	parser.add_argument("--no-write", action="store_true", help="Don't write out generated games at all.")
	parser.add_argument("--supervised", metavar="CMD", default=None, type=str, help="Command for a UAI engine.")
	parser.add_argument("--supervised-ms", metavar="N", default=100, type=int, help="Number of milliseconds per move for supervised generation.")
	args = parser.parse_args()

	network_path = args.network
	network_name = os.path.splitext(os.path.basename(args.network))

	if args.random_play:
		print("Doing random play! Loading no model, and not using RPC.")
	elif args.use_rpc:
		engine.setup_evaluator(use_rpc=True, temperature=LOGIT_TEMPERATURE)
	else:
		engine.setup_evaluator(use_rpc=False, temperature=LOGIT_TEMPERATURE)
		engine.initialize_model(network_path)

	if args.supervised != None:
		args.uai_player = uai_ringmaster.UAIPlayer(args.supervised)

	if args.output_games == None:
		output_directory = os.path.join("games", network_name)
		if not os.path.exists(output_directory):
			os.mkdir(output_directory)
		output_path = os.path.join(output_directory, os.urandom(8).encode("hex") + ".json")
	else:
		output_path = args.output_games
	if args.no_write:
		output_path = "/dev/null"
	print("[%3i] Writing to: %s" % (args.group_index, output_path))

	with open(output_path, "w") as f:
		games_generated = 0
		while True:
			entry = generate_game(args)
			if entry["result"] is None:
				print("[%3i] Skipping game with null result." % (args.group_index,))
				continue
			json.dump(entry, f)
			f.write("\n")
			f.flush()
			games_generated += 1
			if args.game_count != None and games_generated >= args.game_count:
				print("Done generating games.")
				break

