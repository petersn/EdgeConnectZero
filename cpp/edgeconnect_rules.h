
#ifndef EDGECONNECT_RULES_H
#define EDGECONNECT_RULES_H

#include <cassert>
#include <unordered_map>
#include <set>
#include <array>
#include <iostream>
#include <vector>
#include <cstdint>

#include "union_find.h"

template<int s1, int s2, int s3>
int stride_index(int x, int y, int z) {
	assert(0 <= x and x < s1);
	assert(0 <= y and y < s2);
	assert(0 <= z and z < s3);
	return
		s2 * s3 * x +
		     s3 * y +
			      z;
}

constexpr int STARTING_GAME_POSITION = 1234;

constexpr int BOARD_RADIUS = 3;
constexpr int BOARD_SIZE = 2 * BOARD_RADIUS + 1;
constexpr int QR_COUNT = BOARD_SIZE * BOARD_SIZE;
constexpr int FEATURE_COUNT = 12;
constexpr int FEATURE_MAP_LENGTH = BOARD_SIZE * BOARD_SIZE * FEATURE_COUNT;

typedef uint8_t Cell;
typedef uint16_t Move;
constexpr Move NO_MOVE = 0xffff;

enum Side {
	NOBODY = 0,
	FIRST_PLAYER = 1,
	SECOND_PLAYER = 2,
};

static inline Move pack_qr(int q, int r) {
	return q + r * BOARD_SIZE;
}

Cell& get_at(std::array<Cell, QR_COUNT>& board, int q, int r) {
	return board[q + r * BOARD_SIZE];
}

const Cell& get_at(const std::array<Cell, QR_COUNT>& board, int q, int r) {
	return board[q + r * BOARD_SIZE];
}

std::array<Cell, QR_COUNT> VALID_CELLS_MASK;
std::array<Cell, QR_COUNT> SCORING_CELLS_MASK;
std::array<Cell, QR_COUNT> EDGE_CELLS_MASK;
std::vector<std::vector<Move>> QR_NEIGHBORS(QR_COUNT);

bool coord_is_valid(int q, int r) {
	if (not (0 <= q and q < BOARD_SIZE and 0 <= r and r < BOARD_SIZE))
		return false;
	return get_at(VALID_CELLS_MASK, q, r);
}

void debug_print(const std::array<Cell, QR_COUNT>& a) {
	for (int y = 0; y < BOARD_SIZE; y++) {
		for (int x = 0; x < BOARD_SIZE; x++) {
			std::cout << " " << (get_at(a, x, y) ? '#' : '.' );
		}
		std::cout << std::endl;
	}
}

static void edgeconnect_initialize_structures() {
	// Compute valid cells.
	for (int packed_qr = 0; packed_qr < QR_COUNT; packed_qr++)
		VALID_CELLS_MASK[packed_qr] = 1;
	for (int i = 0; i < BOARD_RADIUS; i++) {
		for (int j = 0; j < BOARD_RADIUS; j++) {
			if (i + j >= BOARD_RADIUS)
				continue;
			get_at(VALID_CELLS_MASK, i, j) = 0;
			get_at(VALID_CELLS_MASK, BOARD_SIZE - i - 1, BOARD_SIZE - j - 1) = 0;
		}
	}

	for (int packed_qr = 0; packed_qr < QR_COUNT; packed_qr++)
		SCORING_CELLS_MASK[packed_qr] = 0;
	for (int i = 0; i < BOARD_SIZE; i++) {
		get_at(SCORING_CELLS_MASK, i,              0) = 1;
		get_at(SCORING_CELLS_MASK, i, BOARD_SIZE - 1) = 1;
		get_at(SCORING_CELLS_MASK,              0, i) = 1;
		get_at(SCORING_CELLS_MASK, BOARD_SIZE - 1, i) = 1;
	}
	for (int i = 0; i < BOARD_SIZE; i++) {
		get_at(SCORING_CELLS_MASK, i, BOARD_RADIUS - i) = 1;
		get_at(SCORING_CELLS_MASK, BOARD_SIZE - i - 1, BOARD_SIZE - (BOARD_RADIUS - i) - 1) = 1;
	}
	// Mask down to just the valid positions.
	for (int packed_qr = 0; packed_qr < QR_COUNT; packed_qr++)
		SCORING_CELLS_MASK[packed_qr] *= VALID_CELLS_MASK[packed_qr];
	std::copy(SCORING_CELLS_MASK.begin(), SCORING_CELLS_MASK.end(), EDGE_CELLS_MASK.begin());
	get_at(SCORING_CELLS_MASK, BOARD_RADIUS, BOARD_RADIUS) = 1;

	std::cout << "VALID:\n";
	debug_print(VALID_CELLS_MASK);
	std::cout << "SCORING:\n";
	debug_print(SCORING_CELLS_MASK);
	std::cout << "EDGE:\n";
	debug_print(EDGE_CELLS_MASK);

	for (int q = 0; q < BOARD_SIZE; q++) {
		for (int r = 0; r < BOARD_SIZE; r++) {
			if (not coord_is_valid(q, r))
				continue;
			for (auto offset : {
				std::pair<int, int>
				{+1,  0}, {-1,  0},
				{ 0, +1}, { 0, -1},
				{+1, -1}, {-1, +1},
			}) {
				int qo = q + offset.first, ro = r + offset.second;
				if (coord_is_valid(qo, ro))
					QR_NEIGHBORS[pack_qr(q, r)].push_back(pack_qr(qo, ro));
			}
		}
	}
};

static UnionFind<Move> compute_union_find(const std::array<Cell, QR_COUNT>& cells) {
	UnionFind<Move> uf;
	for (Move m = 0; m < QR_COUNT; m++)
		if (VALID_CELLS_MASK[m])
			for (Move neighbor : QR_NEIGHBORS[m])
				if (cells[m] == cells[neighbor])
					uf.union_nodes(m, neighbor);
	return uf;
}

struct EdgeConnectState {
	std::array<Cell, QR_COUNT> cells{};
	int move_state = 1;
	Move first_move_qr = NO_MOVE;

	int legal_moves(Move* output_buffer) const {
		int count = 0;
		for (Move m = 0; m < QR_COUNT; m++) {
			if (VALID_CELLS_MASK[m] and cells[m] == 0) {
				*output_buffer++ = m;
				count++;
			}
		}
		return count;
	}

	Side get_side() const {
		return move_state <= 1 ? Side::FIRST_PLAYER : Side::SECOND_PLAYER;
	}

	void make_move(Move m) {
		assert(VALID_CELLS_MASK[m]);
		assert(cells[m] == 0);
		cells[m] = move_state <= 1 ? 1 : 2;
		if (move_state % 2 == 0)
			first_move_qr = m;
		move_state++;
		move_state %= 4;
	}

	Side result() const {
		// Check for free cells.
		for (Move m = 0; m < QR_COUNT; m++)
			if (VALID_CELLS_MASK[m] and cells[m] == 0)
				return Side::NOBODY;

		// Make a copy of the cells to do captures in.
		UnionFind<Move> uf = compute_union_find(cells);
		std::unordered_map<UnionFind<Move>::NodeIndex, int> group_edge_count;
		for (Move m = 0; m < QR_COUNT; m++)
			if (EDGE_CELLS_MASK[m])
				group_edge_count[uf.find(m)]++;
		std::array<Cell, QR_COUNT> cells_with_captures = cells;
		for (Move m = 0; m < QR_COUNT; m++)
			if (VALID_CELLS_MASK[m] and group_edge_count[uf.find(m)] <= 1)
				cells_with_captures[m] = 3 - cells_with_captures[m];

		// Compute group based adjustments.
		UnionFind<Move> uf_with_captures = compute_union_find(cells_with_captures);
		std::unordered_map<int, std::set<UnionFind<Move>::NodeIndex>> groups;
		for (Move m = 0; m < QR_COUNT; m++)
			if (VALID_CELLS_MASK[m])
				groups[cells_with_captures[m]].insert(uf_with_captures.find(m));

		// Group counts.
		int first_player_group_count = groups[Side::FIRST_PLAYER].size();
		int second_player_group_count = groups[Side::SECOND_PLAYER].size();

//		std::cout << "Group counts: " << first_player_group_count << " " << second_player_group_count << std::endl;

		int player_scores[2] = {};
		// Count who has more points.
		for (Move m = 0; m < QR_COUNT; m++)
			if (SCORING_CELLS_MASK[m])
				player_scores[cells_with_captures[m] - 1]++;

		player_scores[0] += 2 * (second_player_group_count - first_player_group_count);
		player_scores[1] += 2 * (first_player_group_count - second_player_group_count);

		assert(player_scores[0] != player_scores[1]);
		return player_scores[0] > player_scores[1] ? Side::FIRST_PLAYER : Side::SECOND_PLAYER;
	}

	Side result_with_early_stopping() const {
		Side results[2];
		for (Side player : {Side::FIRST_PLAYER, Side::SECOND_PLAYER}) {
			EdgeConnectState filled_in_for_player = *this;
			for (Move m = 0; m < QR_COUNT; m++)
				if (VALID_CELLS_MASK[m] and filled_in_for_player.cells[m] == 0)
					filled_in_for_player.cells[m] = player;
			results[player - 1] = filled_in_for_player.result();
		}
		if (results[0] == results[1])
			return results[0];
		return Side::NOBODY;
	}

	void featurize(float* feature_buffer) const {
		// Make a copy of the cells to do captures in.
		UnionFind<Move> uf = compute_union_find(cells);
		std::unordered_map<UnionFind<Move>::NodeIndex, int> group_edge_count;
		for (Move m = 0; m < QR_COUNT; m++)
			if (EDGE_CELLS_MASK[m])
				group_edge_count[uf.find(m)]++;
		for (int q = 0; q < BOARD_SIZE; q++) {
			for (int r = 0; r < BOARD_SIZE; r++) {	
				// Layer 0: All ones.
				feature_buffer[stride_index<BOARD_SIZE, BOARD_SIZE, FEATURE_COUNT>(q, r, 0)] = 1;
				// Layer 1: In bound cells.
				feature_buffer[stride_index<BOARD_SIZE, BOARD_SIZE, FEATURE_COUNT>(q, r, 1)] = coord_is_valid(q, r);
				// Layer 2: Scoring cells.
				feature_buffer[stride_index<BOARD_SIZE, BOARD_SIZE, FEATURE_COUNT>(q, r, 2)] = get_at(SCORING_CELLS_MASK, q, r);
				// Layer 3: Edge cells.
				feature_buffer[stride_index<BOARD_SIZE, BOARD_SIZE, FEATURE_COUNT>(q, r, 3)] = get_at(EDGE_CELLS_MASK, q, r);
				// Layers 4 and 5: On our first or second move.
				feature_buffer[stride_index<BOARD_SIZE, BOARD_SIZE, FEATURE_COUNT>(q, r, 4)] = move_state % 2 == 0;
				feature_buffer[stride_index<BOARD_SIZE, BOARD_SIZE, FEATURE_COUNT>(q, r, 5)] = move_state % 2 == 1;
				// Layer 6: Our stones.
				feature_buffer[stride_index<BOARD_SIZE, BOARD_SIZE, FEATURE_COUNT>(q, r, 6)] = get_at(cells, q, r) == get_side();
				// Layer 7: Their stones.
				feature_buffer[stride_index<BOARD_SIZE, BOARD_SIZE, FEATURE_COUNT>(q, r, 7)] = get_at(cells, q, r) == 3 - get_side();
				// Layer 8: Our last move, if we're on our second move.
				feature_buffer[stride_index<BOARD_SIZE, BOARD_SIZE, FEATURE_COUNT>(q, r, 8)] = pack_qr(q, r) == first_move_qr;
				// Layer 9, 10, 11: Groups with no edge cells, one edge cell, and two or more edge cells.
				int edge_count = group_edge_count[uf.find(pack_qr(q, r))];
				feature_buffer[stride_index<BOARD_SIZE, BOARD_SIZE, FEATURE_COUNT>(q, r, 9)] = edge_count == 0 and get_at(cells, q, r) != 0;
				feature_buffer[stride_index<BOARD_SIZE, BOARD_SIZE, FEATURE_COUNT>(q, r, 10)] = edge_count == 1 and get_at(cells, q, r) != 0;
				feature_buffer[stride_index<BOARD_SIZE, BOARD_SIZE, FEATURE_COUNT>(q, r, 11)] = edge_count >= 2 and get_at(cells, q, r) != 0;
			}
		}
	}
};

#endif

