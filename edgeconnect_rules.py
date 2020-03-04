#!/usr/bin/python

import functools
import itertools
import numpy as np
import UnionFind

RED  = "\x1b[91m"
BLUE = "\x1b[94m"
ENDC = "\x1b[0m"

BOARD_RADIUS = 11
BOARD_SIZE = 2 * BOARD_RADIUS + 1

NEXT_MOVE_STATE = {
	(1, 'a'): (1, 'b'),
	(1, 'b'): (2, 'a'),
	(2, 'a'): (2, 'b'),
	(2, 'b'): (1, 'a'),
}
OTHER_PLAYER = {1: 2, 2: 1}

VALID_CELLS_MASK = np.ones((BOARD_SIZE, BOARD_SIZE), np.int8)
for i in range(BOARD_RADIUS):
	for j in range(BOARD_RADIUS):
		if i + j >= BOARD_RADIUS:
			continue
		VALID_CELLS_MASK[i, j] = 0
		VALID_CELLS_MASK[-i - 1, -j - 1] = 0
ALL_VALID_QR = [qr for qr, mask in np.ndenumerate(VALID_CELLS_MASK) if mask]

SCORING_CELLS_MASK = np.zeros((BOARD_SIZE, BOARD_SIZE), np.int8)
SCORING_CELLS_MASK[(0, -1), :] = 1
SCORING_CELLS_MASK[:, (0, -1)] = 1
for i in range(BOARD_SIZE):
	SCORING_CELLS_MASK[i, BOARD_RADIUS - i] = 1
	SCORING_CELLS_MASK[-i - 1, -(BOARD_RADIUS - i) - 1] = 1
# Compute the edge cells before we add in the final middle scoring cell.
EDGE_CELLS_MASK = SCORING_CELLS_MASK * VALID_CELLS_MASK
ALL_EDGE_QR = [qr for qr, mask in np.ndenumerate(SCORING_CELLS_MASK) if mask]
SCORING_CELLS_MASK[BOARD_RADIUS, BOARD_RADIUS] = 1
SCORING_CELLS_MASK *= VALID_CELLS_MASK

QR_NEIGHBOR_OFFSETS = [
	(+1,  0), (-1,  0),
	( 0, +1), ( 0, -1),
	(+1, -1), (-1, +1),
]
QR_NEIGHBORS = {}
for qr in ALL_VALID_QR:
	QR_NEIGHBORS[qr] = []
	for offset in QR_NEIGHBOR_OFFSETS:
		offset_qr = qr[0] + offset[0], qr[1] + offset[1]
		if offset_qr in ALL_VALID_QR:
			QR_NEIGHBORS[qr].append(offset_qr)

svg_template = """
<svg width="{width}" height="{height}">
	<defs>
		<pattern id="grid" width="1" height="0.8660254" patternUnits="userSpaceOnUse">
			<path d="M 1 0 L 0 0 0 0.8660254" fill="none" stroke="gray" stroke-width="0.1"/>
		</pattern>
	</defs>

	<g transform="translate(0, 2)">
		<g transform="scale(20)">
		{objects}
		</g>
	</g>
</svg>
"""

background_grid = """<rect width="100%" height="100%" fill="url(#grid)"></rect>"""
#blue_circle = """<circle cx="0.5" cy="0.433" r="0.45" stroke="black" stroke-width="0.05" fill="blue"></circle>"""
#red_circle  = """<circle cx="0.5" cy="0.433" r="0.45" stroke="black" stroke-width="0.05" fill="red"></circle>"""
#grey_circle = """<circle cx="0.5" cy="0.433" r="0.45" stroke="black" stroke-width="0.05" fill="grey"></circle>"""

blue_circle = """
<path d="M 0 0 l 0 0.5  0.5 0.36  0.5 -0.36  0 -0.5  -0.5 -0.36 Z" stroke="black" stroke-width="0.05" fill="blue"></path>
"""

red_circle = """
<path d="M 0 0 l 0 0.5  0.5 0.36  0.5 -0.36  0 -0.5  -0.5 -0.36 Z" stroke="black" stroke-width="0.05" fill="red"></path>
"""

light_blue_circle = """
<path d="M 0 0 l 0 0.5  0.5 0.36  0.5 -0.36  0 -0.5  -0.5 -0.36 Z" stroke="black" stroke-width="0.05" fill="#88f"></path>
"""

light_red_circle = """
<path d="M 0 0 l 0 0.5  0.5 0.36  0.5 -0.36  0 -0.5  -0.5 -0.36 Z" stroke="black" stroke-width="0.05" fill="#f88"></path>
"""

suggested_blue = """
<path d="M 0 0 l 0 0.5  0.5 0.36  0.5 -0.36  0 -0.5  -0.5 -0.36 Z" stroke="blue" stroke-width="0.1" fill="darkgrey"></path>
"""

suggested_red = """
<path d="M 0 0 l 0 0.5  0.5 0.36  0.5 -0.36  0 -0.5  -0.5 -0.36 Z" stroke="red" stroke-width="0.1" fill="darkgrey"></path>
"""


grey_circle = """
<path d="M 0 0 l 0 0.5  0.5 0.36  0.5 -0.36  0 -0.5  -0.5 -0.36 Z" stroke="black" stroke-width="0.05" fill="darkgrey"></path>
"""

"""<circle cx="0.5" cy="0.433" r="0.45" stroke="black" stroke-width="0.05" fill="blue"></circle>"""
#red_circle  = """<circle cx="0.5" cy="0.433" r="0.45" stroke="black" stroke-width="0.05" fill="red"></circle>"""
#grey_circle = """<circle cx="0.5" cy="0.433" r="0.45" stroke="black" stroke-width="0.05" fill="grey"></circle>"""

funky_constant = 1.5 / 3**0.5

def translate_svg(x, y, inner_svg):
	return """
		<g transform="translate({}, {})">
			{}
		</g>
	""".format(x + 0.025, y + 0.3, inner_svg)

def qr_to_xy(qr):
	return np.array([
		[1,            0.5],
		[0, funky_constant],
	]).dot(qr)

def small_rotate_board(board, direction):
	new = np.zeros_like(board)
	for original_qr in ALL_VALID_QR:
		qr = np.array(original_qr) - BOARD_RADIUS
		if direction:
			qr = np.array([[1, 1], [-1, 0]]).dot(qr)
		else:
			qr = np.array([[0, -1], [1, 1]]).dot(qr)
		qr += BOARD_RADIUS
		new[tuple(qr)] = board[original_qr]
	return new

@functools.lru_cache(maxsize=1000000)
def apply_symmetry_to_qr(symmetry, qr):
	assert 0 <= symmetry < 12
	do_flip  = symmetry >= 6
	rotation = symmetry % 6
	if do_flip:
		qr = qr[::-1]
	qr = np.array(qr) - BOARD_RADIUS
	for _ in range(rotation):
		qr = np.array([[1, 1], [-1, 0]]).dot(qr)
	qr += BOARD_RADIUS
	return tuple(qr)

def apply_symmetry_to_board(symmetry, board):
	new = np.zeros_like(board)
	for qr in ALL_VALID_QR:
		new[apply_symmetry_to_qr(symmetry, qr)] = board[qr]
	return new

class EdgeConnectState:
	def __init__(self, board, move_state=(1, 'b'), first_move_qr=None, legal_moves_cache=None):
		self.board = board
		self.move_state = move_state
		self.first_move_qr = first_move_qr
		self.legal_moves_cache = legal_moves_cache
		assert move_state in NEXT_MOVE_STATE
#		assert (first_move_qr == None) == (move_state[1] == 'a')

	@staticmethod
	def initial():
		return EdgeConnectState(np.zeros((BOARD_SIZE, BOARD_SIZE), np.int8))

	def to_string(self):
		return "%i%s%s-%s" % (
			self.move_state[0],
			self.move_state[1],
			"".join(str(c) for qr, c in np.ndenumerate(self.board) if VALID_CELLS_MASK[qr]),
			"0-0" if self.first_move_qr is None else "%i-%i" % self.first_move_qr,
		)

	@staticmethod
	def from_string(serialization):
		state = EdgeConnectState.initial()
		state.move_state = int(serialization[0]), serialization[1]
		assert state.move_state in NEXT_MOVE_STATE
		if serialization.count("-") == 0:
			board_desc, fm_q, fm_r = serialization[2:], "0", "0"
		else:
			board_desc, fm_q, fm_r = serialization[2:].split("-")
		assert len(board_desc) == len(ALL_VALID_QR)
		for c, qr in zip(board_desc, ALL_VALID_QR):
			state.board[qr] = int(c)
			assert c in ("0", "1", "2")
		if (fm_q, fm_r) == ("0", "0"):
			state.first_move_qr = None
		else:
			state.first_move_qr = int(fm_q), int(fm_r)
		return state

	def sanity_check(self):
		if self.move_state[1] == "b":
			if self.first_move_qr is None:
				pass
#				if not np.all(self.board == 0):
#					print("FAIL BEGINNING " * 100)
#					return False
			elif self.board[self.first_move_qr] != self.move_state[0]:
				print("FAIL CLEAR " * 100)
				return False
		# TODO: Add more checks here.
		return True

	def _repr_svg_(self):
		pixel_width = 1 + BOARD_SIZE * 20
		pixel_height = 1 + BOARD_SIZE * funky_constant * 20 + 8
		#objects = [background_grid]
		objects = []
		for z_pass in (False, True):
			for qr, mask in np.ndenumerate(VALID_CELLS_MASK):
				if mask == 0:
					assert self.board[qr] == 0
					continue
				x, y = qr_to_xy(qr)
				if (self.board[qr] in (5, 6)) != z_pass:
					continue
				objects.append(translate_svg(x - BOARD_RADIUS / 2.0, y, {
					0: grey_circle,
					1: red_circle,
					2: blue_circle,
					3: light_red_circle,
					4: light_blue_circle,
					5: suggested_red,
					6: suggested_blue,
				}[self.board[qr]]))
		return svg_template.format(
			width=pixel_width,
			height=pixel_height,
			objects="\n".join(objects),
		)

	def __str__(self):
		return "\n".join(
			(" " * y + " ".join(
				{
					0: ("o" if SCORING_CELLS_MASK[x, y] else ".")
						if VALID_CELLS_MASK[x, y] else " ",
					1: RED + "X" + ENDC,
					2: BLUE + "O" + ENDC,
				}[self.board[x, y]]
				for x in range(BOARD_SIZE)
			))[BOARD_RADIUS:]
			for y in range(BOARD_SIZE)
		)


	def copy(self):
		return EdgeConnectState(
			board=self.board.copy(),
			move_state=self.move_state,
			first_move_qr=self.first_move_qr,
		)

	def legal_moves(self):
		if self.legal_moves_cache is None:
			self.legal_moves_cache = [
				qr for qr in ALL_VALID_QR
				if self.board[qr] == 0
			]
		return self.legal_moves_cache

	def make_move(self, qr):
		assert qr in ALL_VALID_QR, "Bad containment: %r %r" % (qr, ALL_VALID_QR)
		assert self.board[qr] == 0
		self.legal_moves_cache = None
		self.board[qr] = self.move_state[0]
		self.first_move_qr = qr if self.move_state[1] == 'a' else None
		self.move_state = NEXT_MOVE_STATE[self.move_state]

	def compute_group_union_find(self, board):
		uf = UnionFind.UnionFind()
		for qr in ALL_VALID_QR:
			for neighbor in QR_NEIGHBORS[qr]:
				if board[qr] == board[neighbor]:
					uf.union(qr, neighbor)
		return uf

	def compute_group_counts(self, board=None):
		board = self.board if board is None else board
		uf = self.compute_group_union_find(board)
		groups = {0: set(), 1: set(), 2: set()}
		for qr in ALL_VALID_QR:
			player = board[qr]
			groups[player].add(uf[qr])
		return {player_index: len(group_set) for player_index, group_set in groups.items()}

	def apply_captures(self):
		# TODO: Ask Taras what the right rules are for this.
		board = self.board
		while True:
			uf = self.compute_group_union_find(board)
			# Find each group with only one cell.
			group_edge_count = {}
			for qr in ALL_EDGE_QR:
				canonicalized = uf[qr]
				if canonicalized not in group_edge_count:
					group_edge_count[canonicalized] = 0
				group_edge_count[canonicalized] += 1
			result_board = board.copy()
			made_changes = False
			for qr in ALL_VALID_QR:
				if group_edge_count.get(uf[qr], 0) <= 1:
					result_board[qr] = OTHER_PLAYER[result_board[qr]]
					made_changes = True
			if not made_changes:
				break
			board = result_board
		return result_board

	def compute_scores(self):
		# 1) Evaluate captures.
		captures_board = self.apply_captures()

		# 2) Compute edge scores for each player.
		scores = {
			player: ((captures_board * SCORING_CELLS_MASK) == player).sum()
			for player in (1, 2)
		}

		# 3) Compute group based adjustments.
		group_counts = self.compute_group_counts(captures_board)
		for player in (1, 2):
			scores[player] += 2 * (group_counts[OTHER_PLAYER[player]] - group_counts[player])

		return scores, EdgeConnectState(captures_board)

	def result(self):
		if (self.board != 0).sum() != len(ALL_VALID_QR):
			return
		scores, _ = self.compute_scores()
		assert scores[1] != scores[2]
		return 1 if scores[1] > scores[2] else 2

	def make_early_stopping_boards(self):
		filled_in_boards = {}
		for player in (1, 2):
			duplicate = self.copy()
			for qr in ALL_VALID_QR:
				if duplicate.board[qr] == 0:
					duplicate.board[qr] = player
			filled_in_boards[player] = duplicate
		return filled_in_boards

	def result_with_early_stopping(self):
		optimistic = {k: v.result() for k, v in self.make_early_stopping_boards().items()}
		if optimistic[1] == optimistic[2]:
			return optimistic[1]

	def unconditional_captures(self):
		filled_in_boards = {k: v.apply_captures() for k, v in self.make_early_stopping_boards().items()}
		return EdgeConnectState(np.where(filled_in_boards[1] == filled_in_boards[2], filled_in_boards[1], self.board))

	def featurize_board(self, symmetry):
		symm_board = apply_symmetry_to_board(symmetry, self.board)
		result = np.zeros((BOARD_SIZE, BOARD_SIZE, 12), np.float32)
		# Layer 0: All ones.
		result[:, :, 0] = 1
		# Layer 1: In bound cells.
		result[:, :, 1] = VALID_CELLS_MASK
		# Layer 2: Scoring cells.
		result[:, :, 2] = SCORING_CELLS_MASK
		# Layer 3: Edge cells.
		result[:, :, 3] = EDGE_CELLS_MASK
		# Layers 4 and 5: On our first or second move.
		result[:, :, {'a': 4, 'b': 5}[self.move_state[1]]] = 1
		# Layer 6: Our stones.
		result[:, :, 6] = symm_board == self.move_state[0]
		# Layer 7: Their stones.
		result[:, :, 7] = symm_board == OTHER_PLAYER[self.move_state[0]]
		# Layer 8: Our last move, if we're on our second move.
		if self.first_move_qr is not None:
			fmqr = apply_symmetry_to_qr(symmetry, self.first_move_qr)
			result[fmqr[0], fmqr[1], 8] = 1

		# TODO: Ask Taras what the right rules are for this.
		uf = self.compute_group_union_find(symm_board)
		# Find each group with only one cell.
		group_edge_count = {}
		for qr in ALL_EDGE_QR:
			canonicalized = uf[qr]
			if canonicalized not in group_edge_count:
				group_edge_count[canonicalized] = 0
			group_edge_count[canonicalized] += 1
		group_edge_count_array = np.zeros_like(symm_board)
		for qr in ALL_VALID_QR:
			group_edge_count_array[qr] = group_edge_count.get(uf[qr], 0)
		result[:, :, 9] = (group_edge_count_array == 0) & (symm_board != 0)
		result[:, :, 10] = (group_edge_count_array == 1) & (symm_board != 0)
		result[:, :, 11] = (group_edge_count_array >= 2) & (symm_board != 0)
		return result

if __name__ == "__main__":
	s = EdgeConnectState.initial()
	print("BOARD RADIUS:", BOARD_RADIUS, "VALID QR COUNT:", len(ALL_VALID_QR))

