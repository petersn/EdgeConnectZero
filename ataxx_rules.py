#!/usr/bin/python

import array

RED  = "\x1b[91m"
BLUE = "\x1b[94m"
ENDC = "\x1b[0m"
SIZE = 7
#BLOCKED_CELLS = frozenset([(3, 2), (2, 3), (4, 3), (3, 4)])
BLOCKED_CELLS = frozenset()
LEGAL_SQUARE_COUNT = SIZE * SIZE - len(BLOCKED_CELLS)
OTHER_PLAYER = {1: 2, 2: 1}
NEAR_NEIGHBOR_OFFSETS = [
	(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)
	if (a, b) != (0, 0)
]
FAR_NEIGHBOR_OFFSETS = [
	(a, b) for a in (-2, -1, 0, 1, 2) for b in (-2, -1, 0, 1, 2)
	if (a, b) != (0, 0) and (a, b) not in NEAR_NEIGHBOR_OFFSETS
]

def legal_spot(xy):
	x, y = xy
	return xy not in BLOCKED_CELLS and 0 <= x < SIZE and 0 <= y < SIZE

def get_neighbors(xy, offsets):
	x, y = xy
	return [
		(x + i, y + j)
		for i, j in offsets
		if legal_spot((x + i, y + j))
	]

def Linf_distance(a, b):
	return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

class AtaxxState:
	def __init__(self, board, to_move=1, legal_moves_cache=None):
		self.board = board
		self.to_move = to_move
		self.legal_moves_cache = legal_moves_cache

	@staticmethod
	def initial():
		board = AtaxxState(array.array("b", [0] * (SIZE * SIZE)))
		board[0, 0] = 1
		board[SIZE - 1, SIZE - 1] = 1
		board[SIZE - 1, 0] = 2
		board[0, SIZE - 1] = 2
		return board

	@staticmethod
	def from_fen(fen):
		player_mapping = {"x": 1, "o": 2}
		components = fen.lower().split()
		board_state_string, player_to_move = components[:2]
		board = AtaxxState(
			board=array.array("b", [0] * (SIZE * SIZE)),
			to_move=player_mapping[player_to_move],
		)
		for y, chunk in enumerate(board_state_string.split("/")):
			x = 0
			for c in chunk:
				if c in ("1", "2", "3", "4", "5", "6", "7"):
					x += int(c)
					continue
				board[x, y] = player_mapping[c]
				x += 1
		return board

	def copy(self):
		return AtaxxState(self.board[:], self.to_move, self.legal_moves_cache)

	def __setitem__(self, index, value):
		x, y = index
		self.board[x + y * SIZE] = value

	def __getitem__(self, index):
		x, y = index
		return self.board[x + y * SIZE]

	def __eq__(self, other):
		return self.to_move == other.to_move and self.board == other.board

	def __str__(self):
		return "\n".join(
			" ".join(
				"#" if (x, y) in BLOCKED_CELLS else
				{0: ".", 1: RED + "X" + ENDC, 2: BLUE + "O" + ENDC}[self[x, y]]
				for x in range(SIZE)
			)
			for y in range(SIZE)
		)

	def fen(self):
		s = "/".join(
			"".join(
				"-" if (x, y) in BLOCKED_CELLS else
				{0: ".", 1: "x", 2: "o"}[self[x, y]]
				for x in range(SIZE)
			)
			for y in range(SIZE)
		) + " " + {1: "x", 2: "o"}[self.to_move]
		for i in range(1, SIZE + 1)[::-1]:
			s = s.replace("."*i, str(i))
		return s

	def move(self, desc):
		if hasattr(self, "evaluations"):
			raise Exception("BAD")
		self.legal_moves_cache = None
		if desc == "pass":
			self.to_move = OTHER_PLAYER[self.to_move]
			return
		start, end = desc
		if start != "c":
			assert self[start] == self.to_move
		assert end not in BLOCKED_CELLS
		assert self[end] == 0
		self[end] = self.to_move
		if start != "c":
			distance = Linf_distance(start, end)
			assert distance in (1, 2)
			if distance == 2:
				self[start] = 0
		self.capture_for(self.to_move, end)
		self.to_move = OTHER_PLAYER[self.to_move]

	def capture_for(self, player, xy):
		for neighbor in get_neighbors(xy, offsets=NEAR_NEIGHBOR_OFFSETS):
			if self[neighbor] != 0:
				self[neighbor] = player

	def legal_moves(self):
		if self.legal_moves_cache is None:
			self.legal_moves_cache = []
			can_copy_to = set()
			for x in range(SIZE):
				for y in range(SIZE):
					source = x, y
					# Skip cells that aren't our pieces.
					if self[source] != self.to_move:
						continue
					# Try to find movement moves.
					for dest in get_neighbors(source, offsets=FAR_NEIGHBOR_OFFSETS):
						if self[dest] == 0:
							self.legal_moves_cache.append((source, dest))
					# Find clone moves.
					for dest in get_neighbors(source, offsets=NEAR_NEIGHBOR_OFFSETS):
						if self[dest] == 0:
							can_copy_to.add(("c", dest))
			# Add in copy moves.
			self.legal_moves_cache.extend(can_copy_to)
			# If we have no legal actual moves then we may pass.
			if not self.legal_moves_cache:
				self.legal_moves_cache = ["pass"]
		return self.legal_moves_cache

	def result(self):
		# Get counts of the various kinds of cells.
		counts = {i: self.board.count(i) for i in (0, 1, 2)}
		assert counts[1] != 0 or counts[2] != 0
		# If the current player has no non-pass moves then score immediately.
		if self.legal_moves() == ["pass"]:
			counts[OTHER_PLAYER[self.to_move]] += counts[0] - len(BLOCKED_CELLS)
			counts[0] = len(BLOCKED_CELLS)
			return max(counts, key=counts.__getitem__)
		# If either player has no pieces then the other player wins.
		if counts[1] == 0:
			return 2
		if counts[2] == 0:
			return 1
		# Check the number of unoccupied squares.
		assert counts[0] >= len(BLOCKED_CELLS)
		if counts[0] != len(BLOCKED_CELLS):
			return None
		# Okay, we're full. Determine who won.
		player_scores = {i: self.board.count(i) for i in (1, 2)}
		return max(player_scores, key=player_scores.__getitem__)

if __name__ == "__main__":
	import random
	print("Doing random play demonstration.")
	state = AtaxxState.initial()
	while True:
		move = random.choice(state.legal_moves())
		state.move(move)
#		print()
		print(move)
#		print(state)
		print(state.fen())
#		print(state.board, state.to_move, state.fen())
		assert AtaxxState.from_fen(state.fen()) == state
		r = state.result()
		if r != None:
			print("Result:", r)
			break

