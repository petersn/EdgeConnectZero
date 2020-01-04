#!/usr/bin/python

import os, glob, json, random, queue, threading, multiprocessing
import tensorflow as tf
import numpy as np
#import ataxx_rules
import edgeconnect_rules
import uai_interface
import engine
import model

#generation_mode = "PLAIN"
#generation_mode = "THREADS"
generation_mode = "PROCESSES"

def parse_move(move):
	if isinstance(move, str):
		if "-" in move:
			return tuple(map(int, move.split("-")))
		move = int(move)
		return move // model.BOARD_SIZE, move % model.BOARD_SIZE
	return tuple(move)

# WARNING: Loops infinitely if there are no games with no non-passing moves.
def get_sample_from_entries(entries):
	while True:
		entry = random.choice(entries)
		ply = random.randrange(len(entry["boards"]))
		if "random_ply" in entry:
			assert False, "I don't know that this code path works!"
			# Note that we need the +1 because we want to train on the board state just AFTER the random move was performed.
			ply = entry["random_ply"] + 1
#		to_move = 1 if ply % 2 == 0 else 2
		#board = ataxx_rules.AtaxxState(entry["boards"][ply], to_move=to_move).copy()
		board = edgeconnect_rules.EdgeConnectState.from_string(entry["boards"][ply])
		assert board.sanity_check()
		move  = entry["moves"][ply]
		if move == "pass":
			continue
		# Convert the board into encoded features.
		symmetry = random.randrange(12)
		features = board.featurize_board(symmetry) #engine.board_to_features(board)
		assert entry["result"] in (1, 2)
		assert board.move_state[0] in (1, 2)

		# The entry["evals"] ranges from 0 to 1, and is the win rate for
		# the current player. We need to remap this to -1 to 1.
		scored_game_value = 1 if entry["result"] == board.move_state[0] else -1
		if "evals" in entry:
			mcts_value = entry["evals"][ply] * 2 - 1
			desired_value = [0.5 * mcts_value + 0.5 * scored_game_value]
		else:
			desired_value = [scored_game_value]
		# Apply a dihedral symmetry.
#		symmetry_index = random.randrange(8)
#		features = apply_symmetry(symmetry_index, features)
		# Build up a map of the desired result.
		desired_policy = np.zeros(
			(model.BOARD_SIZE, model.BOARD_SIZE, model.MOVE_TYPES),
			dtype=np.float32,
		)
		if "dists" not in entry:
			distribution = {uai_interface.uai_encode_move(move): 1}
		else:
			distribution = entry["dists"][ply]
		for move, probability in distribution.items():
			assert probability != 0
			move = parse_move(move)
			assert board.board[move] == 0
			qr = edgeconnect_rules.apply_symmetry_to_qr(symmetry, move)
			desired_policy[qr] = probability
			# Assert that the location is unoccupied for both players.
			assert features[qr[0], qr[1], 6] == 0
			assert features[qr[0], qr[1], 7] == 0
#			move = apply_symmetry_to_move(symmetry_index, move)
#			engine.add_move_to_heatmap(desired_policy, move, probability)
		assert abs(1 - desired_policy.sum()) < 1e-3
#		desired_policy = engine.encode_move_as_heatmap(move)
		return features, desired_policy, desired_value

def load_entries(paths):
	entries = []
	for path in paths:
		with open(path) as f:
			for line in f:
				line = line.strip()
				if not line:
					continue
				entries.append(json.loads(line))
	# Double check that all of the entries are good.
	print("Verifying", len(entries), "entries.")
	for entry in entries:
		board = edgeconnect_rules.EdgeConnectState.from_string(entry["boards"][-1])
		move = parse_move(entry["moves"][-1])
		board.make_move(move)
		assert entry["result"] in (1, 2)
#		print("Final:", board.to_string())
#		print("our result:", board.result(), "theirs:", entry["result"])
		assert board.result_with_early_stopping() == entry["result"]
	random.shuffle(entries)
	return entries

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--games", metavar="PATH", required=True, nargs="+", help="Path to .json self-play games files.")
	parser.add_argument("--old-path", metavar="PATH", help="Path for input network.")
	parser.add_argument("--new-path", metavar="PATH", required=True, help="Path for output network.")
	parser.add_argument("--steps", metavar="COUNT", type=int, default=1000, help="Training steps.")
	parser.add_argument("--minibatch-size", metavar="COUNT", type=int, default=256, help="Minibatch size.")
	parser.add_argument("--learning-rate", metavar="LR", type=float, default=0.001, help="Learning rate.")
	args = parser.parse_args()
	print("Arguments:", args)

	# Shuffle the loaded games deterministically.
	random.seed(123456789)
	entries = load_entries(args.games)
	ply_count = sum(len(entry["moves"]) for entry in entries)
	print("Found %i games with %i plies." % (len(entries), ply_count))

	test_entries = entries[:10]
	train_entries = entries[10:]

	network = model.Network("training_net/", build_training=True)
	sess = tf.InteractiveSession()
	sess.run(tf.initialize_all_variables())
	model.sess = sess

	if args.old_path != None:
		print("Loading old model.")
		model.load_model(network, args.old_path)
	else:
		print("WARNING: Not loading a previous model!")

	def make_minibatch(entries, size):
		batch = {"features": [], "policies": [], "values": []}
		for _ in range(size):
			feature, policy, value = get_sample_from_entries(entries)
			batch["features"].append(feature)
			batch["policies"].append(policy)
			batch["values"].append(value)
		return batch

	# Choose the test set deterministically.
	random.seed(123456789)
	in_sample_val_set = make_minibatch(test_entries, 1024)

	print()
	print("Model dimensions: %i filters, %i blocks, %i parameters." % (model.Network.FILTERS, model.Network.BLOCK_COUNT, network.total_parameters))
	print("Have %i augmented samples, and sampling %i in total." % (ply_count * 12, args.steps * args.minibatch_size))
	print("=== BEGINNING TRAINING ===")

	if generation_mode == "THREADS":
		minibatch_queue = queue.Queue()
		def worker_thread():
			print("Starting worker thread.")
			while True:
				if minibatch_queue.qsize() > 100:
					print("Sleeping.")
					time.sleep(0.5)
				minibatch = make_minibatch(train_entries, args.minibatch_size)
				minibatch_queue.put(minibatch)
		worker_thread = threading.Thread(target=worker_thread)
		worker_thread.start()
	elif generation_mode == "PROCESSES":
		minibatch_queue = multiprocessing.Queue()
		def worker_process(seed, m_queue, size):
			random.seed(seed)
			while True:
				if m_queue.qsize() > 100:
					time.sleep(0.1)
				minibatch = make_minibatch(train_entries, size)
				m_queue.put(minibatch)
		processes = []
		for _ in range(4):
			p = multiprocessing.Process(
				target=worker_process,
				args=(random.getrandbits(64), minibatch_queue, args.minibatch_size),
			)
			p.daemon = True
			p.start()
			processes.append(p)
	else:
		assert generation_mode == "PLAIN"

	# Begin training.
	for step_number in range(args.steps):
		if step_number % 100 == 0:
			policy_loss = network.run_on_samples(network.policy_loss.eval, in_sample_val_set)
			value_loss  = network.run_on_samples(network.value_loss.eval, in_sample_val_set)
#			loss = network.get_loss(in_sample_val_set)
			print("Step: %4i -- loss: %.6f  (policy: %.6f  value: %.6f)" % (
				step_number,
				policy_loss + value_loss,
				policy_loss,
				value_loss,
			))
		if generation_mode == "PLAIN":
			minibatch = make_minibatch(train_entries, args.minibatch_size)
		else:
			minibatch = minibatch_queue.get()
		network.train(minibatch, learning_rate=args.learning_rate)

	# Write out the trained model.
	model.save_model(network, args.new_path)

