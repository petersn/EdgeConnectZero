#!/usr/bin/python

import random
import threading
import time
import numpy as np
import ctypes

dll = ctypes.CDLL("./cpp/mcts.so")

exploration_parameter = ctypes.c_double.in_dll(dll, "exploration_parameter")

initialize = dll.initialize
initialize.restype = None
initialize.argtypes = []

set_state_from_string = dll.set_state_from_string
set_state_from_string.restype = None
set_state_from_string.argtypes = [ctypes.c_char_p]

get_visits_in_current_tree = dll.get_visits_in_current_tree
get_visits_in_current_tree.restype = ctypes.c_int
get_visits_in_current_tree.argtypes = []

think = dll.think
think.restype = ctypes.c_int
think.argtypes = []

launch_mcts = dll.launch_mcts
launch_mcts.restype = None
launch_mcts.argtypes = [ctypes.c_int]

#print("EXPLORATION PARAMETER:", exploration_parameter.value)

if __name__ == "__main__":
	step_count = 4000

	state = edgeconnect_rules.EdgeConnectState.initial()
	for _ in range(10):
		state.make_move(random.choice(state.legal_moves()))

	initialize()
	set_state_from_string(state.to_string().encode("ascii"))
	move_seq = []
	start = time.time()
	for _ in range(step_count):
		move_seq.append(think())
	end = time.time()
	print("Elapsed:", end - start, "Speed:", step_count / (end - start))
	#print(move_seq)

	exit()

	step_count = 4000
	#step_count = 72
	#step_count = 500

	launch_mcts(3)

	for _ in range(1):
		print()
		start = time.time()
		launch_mcts(step_count)
		end = time.time()
		print("Elapsed:", end - start, "Speed:", step_count / (end - start))

