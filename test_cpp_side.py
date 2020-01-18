#!/usr/bin/python

import hashlib
#from tqdm import tqdm
import numpy as np
import random
import edgeconnect_rules
import cpp.edgeconnect_rules as cpp_edgeconnect_rules

tqdm = lambda x: x

random.seed(1)

plane_names = [
	"ones",
	"valid",
	"scoring",
	"edges",
	"first", "second",
	"ours", "theirs",
	"last-move",
	"edge=0", "edge=1", "edge=2",
]
assert len(plane_names) == 12

def feature_hash(array):
	return hashlib.sha256(array.tostring()).hexdigest()

def get_legal_moves(cpp_board):
	moves_array = np.zeros(1024, dtype=np.uint16)
	move_count = cpp_board.legal_moves(cpp_edgeconnect_rules.size_t_to_Move_ptr_helper(moves_array.ctypes.data))
	assert move_count <= len(moves_array)
	return [int(m) for m in moves_array[:move_count]]

def get_features(cpp_board):
	features = np.zeros(cpp_edgeconnect_rules.FEATURE_MAP_LENGTH, np.float32)
	cpp_board.featurize(cpp_edgeconnect_rules.size_t_to_float_ptr_helper(features.ctypes.data))
	return features

def assert_equal(x, y):
	if x != y:
		print("ERROR:", x, "!=", y)

result_remapping = {None: 0, 1: 1, 2: 2}

def unpack_qr(packed_qr):
	p = cpp_edgeconnect_rules.unpack_qr(packed_qr)
	return p.first, p.second

#def unpack_qr(packed_qr):
#	return packed_qr // edgeconnect_rules.BOARD_SIZE, packed_qr % edgeconnect_rules.BOARD_SIZE

#p = cpp_edgeconnect_rules.unpack_qr(20)
#print(p.first, p.second)

def assert_synchronized(board, cpp_board):
	assert_equal(board.to_string(), cpp_board.serialize_board())
	assert_equal(result_remapping[board.result()], cpp_board.result())
	assert_equal(result_remapping[board.result_with_early_stopping()], cpp_board.result_with_early_stopping())
	cpp_legal_moves = [unpack_qr(move) for move in get_legal_moves(cpp_board)]
	assert_equal(cpp_legal_moves, board.legal_moves())

	# Test the featurizations.
	reference_features = board.featurize_board(0)
	cpp_features = get_features(cpp_board).reshape(reference_features.shape)
#	cpp_features = 
#	cpp_features = np.moveaxis(cpp_features, 1, 0)

	if not np.all(reference_features == cpp_features):
		print()
#		print("Reference:")
#		print(np.moveaxis(reference_features, 2, 0))
#		print("Bad:")
#		print(np.moveaxis(cpp_features, 2, 0))
#		print("Diff:")
#		print(np.moveaxis(cpp_features - reference_features, 2, 0))
		for feature_plane in range(reference_features.shape[-1]):
			matching = np.all(reference_features[:, :, feature_plane] == cpp_features[:, :, feature_plane])
			print("Plane %9s match: %2i %5r %2i %2i" % (
				plane_names[feature_plane],
				feature_plane,
				matching,
				reference_features[:, :, feature_plane].sum(),
				cpp_features[:, :, feature_plane].sum(),
			))
			if not matching:
				print("Reference:")
				print(np.moveaxis(reference_features, 2, 0)[feature_plane])
				print("Bad:")
				print(np.moveaxis(cpp_features, 2, 0)[feature_plane])
		exit()

def do_test():
	board = edgeconnect_rules.EdgeConnectState.initial()
	cpp_board = cpp_edgeconnect_rules.EdgeConnectState()

	while True:
		assert_synchronized(board, cpp_board)
		if not board.legal_moves():
			break
		m = random.choice(board.legal_moves())
#		print(">" * 20, m)
		board.make_move(m)
		cpp_board.make_move(cpp_edgeconnect_rules.pack_qr(*m))

if __name__ == "__main__":
	cpp_edgeconnect_rules.edgeconnect_initialize_structures()
	for _ in tqdm(range(100)):
		do_test()

