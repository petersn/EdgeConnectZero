#!/usr/bin/python

import threading
import time
import numpy as np
import ctypes

dll = ctypes.CDLL("./mcts.so")

launch_mcts = dll.launch_mcts
launch_mcts.restype = None
launch_mcts.argtypes = [ctypes.c_int]

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

