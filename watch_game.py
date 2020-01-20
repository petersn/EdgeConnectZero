#!/usr/bin/python3

import sys, json
import edgeconnect_rules

with open(sys.argv[1]) as f:
	lines = f.readlines()

for i, game in enumerate(lines):
	print("===== Game:", i + 1)
	game = json.loads(game)
	for board in game["boards"]:
		print(edgeconnect_rules.EdgeConnectState.from_string(board))
		input("> ")

