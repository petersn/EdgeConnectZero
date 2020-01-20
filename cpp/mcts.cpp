// MCTS engine

#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <list>
#include <queue>
#include <chrono>
#include <random>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <deque>
#include <cmath>
#include <cassert>

#include "json.hpp"
#include "edgeconnect_rules.h"

using std::shared_ptr;
using std::cout;
using std::endl;

constexpr double exploration_parameter = 1.0;
constexpr double dirichlet_alpha = 0.03;
constexpr double dirichlet_weight = 0.25;

std::random_device rd;
std::default_random_engine generator(rd());

// We raise this exception in worker threads when they're done.
struct StopWorking : public std::exception {};

// Configuration.
int global_visits;

struct pair_hash {
public:
	template <typename T, typename U>
	size_t operator()(const std::pair<T, U>& x) const {
		return std::hash<T>()(x.first) + std::hash<U>()(x.second) * 7;
	}
};

std::pair<const float*, double> request_evaluation(int thread_id, const float* feature_string);

struct Evaluations {
	bool game_over;
	double value;
	std::unordered_map<Move, double> posterior;

	void populate(int thread_id, const EdgeConnectState& board, bool use_dirichlet_noise) {
		// Completely reset the evaluation.
		posterior.clear();
		game_over = false;

		int result = board.result();
		if (result != 0) {
			game_over = true;
			// Score from the perspective of the current player.
			assert(result == 1 or result == 2);
			value = result == 1 ? 1.0 : -1.0;
			// Flip the value to be from the perspective of the current player.
			// TODO: XXX: Validate that I got this the right way around.
			if (board.get_side() == Side::SECOND_PLAYER)
				value *= -1;
			return;
		}

		float feature_buffer[FEATURE_MAP_LENGTH] = {};
		board.featurize(feature_buffer);

		auto request_result = request_evaluation(thread_id, feature_buffer);
		const float* posterior_array = request_result.first;
		value = request_result.second;

		// Softmax the posterior array.
		double softmaxed[BOARD_SIZE * BOARD_SIZE];
		for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++)
			softmaxed[i] = exp(posterior_array[i]);
		double total = 0.0;
		for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++)
			total += softmaxed[i];
		if (total != 0.0) {
			for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++)
				softmaxed[i] /= total;
		}

		// Evaluate movegen.
		Move legal_moves[512];
		int num_moves = board.legal_moves(legal_moves);
		double total_probability = 0.0;
		for (int i = 0; i < num_moves; i++) {
			Move move = legal_moves[i];
			double probability = softmaxed[move];
			posterior.insert({move, probability});
			total_probability += probability;
		}
		// Normalize the posterior.
		if (total_probability != 0.0) {
			for (auto& p : posterior)
				p.second /= total_probability;
		}

		assert(num_moves > 0);
		assert(posterior.size() > 0);

		// Add Dirichlet noise.
		if (use_dirichlet_noise) {
			std::gamma_distribution<double> distribution(dirichlet_alpha, 1.0);
			std::vector<double> dirichlet_distribution;
			for (auto& p : posterior)
				dirichlet_distribution.push_back(distribution(generator));
			// Normalize the Dirichlet distribution.
			double total = 0.0;
			for (double x : dirichlet_distribution)
				total += x;
			for (double& x : dirichlet_distribution)
				x /= total;
			// Perform a weighted sum of this Dirichlet distribution and our previous posterior.
			int index = 0;
			for (auto& p : posterior)
				p.second = dirichlet_weight * dirichlet_distribution[index++] + (1.0 - dirichlet_weight) * p.second;
			// Assert approximate normalization.
			double test_total = 0.0;
			for (auto& p : posterior)
				test_total += p.second;
//			assert(0.99 < test_total and test_total < 1.01);
		}
	}
};

struct BoardEvaluator {
	constexpr int MAX_ENSEMBLE_SIZE = 32;
	constexpr int QUEUE_DEPTH = 1024;
	constexpr int PROBABILITY_THRESHOLD = 0.15;
	constexpr int MAX_CACHE_ENTRIES = 10000;

	struct BoardQueue {
		std::mutex queue_mutex;
		std::deque<EdgeConnectState> board_queue;
	};

	std::mutex cache_mutex;
	std::unordered_map<EdgeConnectState, Evaluations> nn_cache;
	std::vector<BoardQueue> per_thread_board_queues;
	std::deque<EdgeConnectState> board_queue;

	BoardEvaluator(int thread_count)
		: per_thread_board_queues(thread_count) {}

	void compute_evaluations(int thread_id, const EdgeConnectState& state, Evaluations& evals) {
		// Unqueue some boards to be computed along with this one.
		std::unique_lock<std::mutex> lk(board_queues_mutex);
	}
};

struct MCTSEdge;
struct MCTSNode;

struct MCTSEdge {
	Move edge_move;
	MCTSNode* parent_node;
	shared_ptr<MCTSNode> child_node;
	double edge_visits = 0;
	double edge_total_score = 0;

	MCTSEdge(Move edge_move, MCTSNode* parent_node, shared_ptr<MCTSNode> child_node)
		: edge_move(edge_move), parent_node(parent_node), child_node(child_node) {}

	double get_edge_score() const {
		if (edge_visits == 0)
			return 0;
		return edge_total_score / edge_visits;
	}

	void adjust_score(double new_score) {
		edge_visits += 1;
		edge_total_score += new_score;
	}
};

struct MCTSNode {
	EdgeConnectState board;
	bool evals_populated = false;
	Evaluations evals;
	int all_edge_visits = 0;
	std::unordered_map<Move, MCTSEdge> outgoing_edges;

	MCTSNode(const EdgeConnectState& board) : board(board) {}

	double total_action_score(const Move& m) {
		assert(evals_populated);
		const auto it = outgoing_edges.find(m);
		double u_score, Q_score;
		if (it == outgoing_edges.end()) {
			u_score = sqrt(1 + all_edge_visits);
			Q_score = 0;
		} else {
			const MCTSEdge& edge = (*it).second;
			u_score = sqrt(1 + all_edge_visits) / (1 + edge.edge_visits);
			Q_score = edge.get_edge_score();
		}
		u_score *= exploration_parameter * evals.posterior.at(m);
		return u_score + Q_score;
	}

	double get_overall_evaluation() {
		double total_score = 0;
		int denominator = 0;
		for (auto& p : outgoing_edges) {
			total_score += p.second.edge_total_score;
			denominator += p.second.edge_visits;
		}
		if (denominator == 0)
			return 0;
		return total_score / denominator;
	}

	void populate_evals(int thread_id, bool use_dirichlet_noise=false) {
		if (evals_populated)
			return;
		evals.populate(thread_id, board, use_dirichlet_noise);
		evals_populated = true;
	}

	Move select_action() {
		assert(evals_populated);
		// If we have no legal moves then return a null move.
		if (evals.posterior.size() == 0)
			return NO_MOVE;
		// If the game is over then return a null move.
		if (evals.game_over)
			return NO_MOVE;
		// Find the best move according to total_action_score.
		Move best_move = NO_MOVE;
		double best_score = -1;
		bool flag = false;
		for (const auto p : evals.posterior) {
			double score = total_action_score(p.first);
			if (p.first == NO_MOVE) {
				std::cerr << "Bad posterior had NO_MOVE in moves list." << endl;
				std::cerr << "Posterior length: " << evals.posterior.size() << endl;
				std::cerr << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
			}
			assert(not std::isnan(score));
			assert(score >= 0.0);
			if (score >= best_score) {
				best_move = p.first;
				best_score = score;
				flag = true;
			}
		}
		if (best_move == NO_MOVE) {
			std::cerr << "                  WARNING Best move is no move!" << endl;
			std::cerr << "       Posterior size: " << evals.posterior.size() << endl;
			std::cerr << "     Flag status: " << flag << " END FLAG" << endl;
		}
		return best_move;
	}
};

namespace std {
	template<> struct hash<EdgeConnectState> {
		size_t operator()(const EdgeConnectState& m) const {
			size_t result = 1234567;
			for (int i = 0; i < QR_COUNT; i++) {
				result ^= (result << 5) + m.cells[i] + (result >> 2);
			}
			result += m.move_state;
			return result;
		}
	};
}

struct MCTS {
	int thread_id;
	EdgeConnectState root_board;
	bool use_dirichlet_noise;
	shared_ptr<MCTSNode> root_node;
	std::unordered_map<EdgeConnectState, std::weak_ptr<MCTSNode>> transposition_table;

	MCTS(int thread_id, const EdgeConnectState& root_board, bool use_dirichlet_noise)
		: thread_id(thread_id), root_board(root_board), use_dirichlet_noise(use_dirichlet_noise)
	{
		init_from_scratch(root_board);
	}

	void init_from_scratch(const EdgeConnectState& root_board) {
		root_node = std::make_shared<MCTSNode>(root_board);
		root_node->populate_evals(thread_id, use_dirichlet_noise);
	}

	shared_ptr<MCTSNode> create_node_with_transposition(const EdgeConnectState& board) {
		auto it = transposition_table.find(board);
		if (it == transposition_table.end()) {
			shared_ptr<MCTSNode> new_node = std::make_shared<MCTSNode>(board);
			transposition_table[board] = new_node;
			return new_node;
		}
		// If the weak reference was collected then clear it out.
		std::weak_ptr<MCTSNode>& cache_entry = (*it).second;
		if (cache_entry.expired()) {
			transposition_table.erase(it);
			// This shouldn't be able to recurse again, because now we'll hit the top code path.
			return create_node_with_transposition(board);
		}
		return std::shared_ptr<MCTSNode>(cache_entry);
	}

	std::tuple<shared_ptr<MCTSNode>, Move, std::vector<MCTSEdge*>> select_principal_variation(bool best=false) {
		shared_ptr<MCTSNode> node = root_node;
		std::vector<MCTSEdge*> edges_on_path;
		Move move = NO_MOVE;
		while (true) {
			if (best) {
				// Pick the edge that has the highest visit count.
				auto it = std::max_element(
					node->outgoing_edges.begin(),
					node->outgoing_edges.end(),
					[](
						const std::pair<Move, MCTSEdge>& a,
						const std::pair<Move, MCTSEdge>& b
					) {
						return a.second.edge_visits < b.second.edge_visits;
					}
				);
				move = (*it).first;
			} else {
				// Pick the edge that has the current highest k-armed bandit value.
				move = node->select_action();
			}
			// If the tree doesn't continue in the direction of this move, then break.
			const auto it = node->outgoing_edges.find(move);
			if (it == node->outgoing_edges.end())
				break;
			MCTSEdge& edge = (*it).second;
			edges_on_path.push_back(&edge);
			node = edge.child_node;
		}
		return std::tuple<shared_ptr<MCTSNode>, Move, std::vector<MCTSEdge*>>{node, move, edges_on_path};
	}

	void step() {
		// 1) Pick a path through the tree.
		auto triple = select_principal_variation();
		// Darn, I wish I had structured bindings already. :(
		shared_ptr<MCTSNode>         leaf_node     = std::get<0>(triple);
		Move                         move          = std::get<1>(triple);
		std::vector<MCTSEdge*>       edges_on_path = std::get<2>(triple);

		shared_ptr<MCTSNode> new_node;

		// 2) If the move is non-null then expand once at the leaf.
		if (move != NO_MOVE) {
			EdgeConnectState new_board = leaf_node->board;
			new_board.make_move(move);
			new_node = create_node_with_transposition(new_board);
//			new_node = std::make_shared<MCTSNode>(new_board);
			auto pair_it_success = leaf_node->outgoing_edges.insert({
				move,
				MCTSEdge{move, leaf_node.get(), new_node},
			});
			MCTSEdge& new_edge = (*pair_it_success.first).second;
			edges_on_path.push_back(&new_edge);
		} else {
			// If the move is null, then we had no legal moves, and we just propagate the score again.
			// This occurs when we're repeatedly hitting a scored final position.
			new_node = leaf_node;
		}

		// 3) Evaluate the new node to get a score to propagate back up the tree.
		new_node->populate_evals(thread_id);
		// Convert the expected value result into a score.
		double value_score = (new_node->evals.value + 1.0) / 2.0;
		// 4) Backup.
		bool inverted = false;
		for (auto it = edges_on_path.rbegin(); it != edges_on_path.rend(); it++) {
			MCTSEdge& edge = **it;
			// Only invert when we're transitioning from a player's second stone to the next player's first.
			if (edge.parent_node->board.move_state % 2 == 1) {
				inverted = not inverted;
				value_score = 1.0 - value_score;
			}
			assert(inverted == (edge.parent_node->board.get_side() != new_node->board.get_side()));
			edge.adjust_score(value_score);
			edge.parent_node->all_edge_visits++;
		}
		if (edges_on_path.size() == 0) {
			cout << ">>> No edges on path!" << endl;
			//print(root_board, true);
			cout << "Valid moves list:" << endl;
			//print_moves(root_board);
			cout << "Select action at root:" << endl;
			Move selected_action = root_node->select_action();
			cout << "Posterior length:" << root_node->evals.posterior.size() << endl;
			cout << "Game over:" << root_node->evals.game_over << endl;
			cout << "Here it is: " << selected_action << endl;
			//cout << "Here it is: " << selected_action.from << " " << selected_action.to << endl;
//			cout << ">>> Move:" << move.from << " " << move.to << endl;
			cout << ">>> Move: " << move << endl;
		}
		assert(edges_on_path.size() != 0);
	}

	void play(Move move) {
		// See if the move is in our root node's outgoing edges.
		auto it = root_node->outgoing_edges.find(move);
		// If we miss, just throw away everything and init from scratch.
		if (it == root_node->outgoing_edges.end()) {
			root_board.make_move(move);
			init_from_scratch(root_board);
			return;
		}
		// Otherwise, reuse a subtree.
		root_node = (*it).second.child_node;
		root_board = root_node->board;
		// Now that a new node is the root we have to redo its evals with Dirichlet noise, if required.
		// This is a little wasteful, when we could just apply Dirichlet noise, but it's not that bad.
		root_node->evals_populated = false;
		root_node->populate_evals(thread_id, use_dirichlet_noise);

	}
};

Move sample_proportionally_to_visits(const shared_ptr<MCTSNode>& node) {
	double x = std::uniform_real_distribution<float>{0, 1}(generator);
	for (const std::pair<Move, MCTSEdge>& p : node->outgoing_edges) {
		double weight = p.second.edge_visits / node->all_edge_visits;
		if (x <= weight)
			return p.first;
		x -= weight;
	}
	// If we somehow slipped through then return some arbitrary element.
	std::cerr << "Potential bug: Weird numerical edge case in sampling!" << endl;
	return (*node->outgoing_edges.begin()).first;
}

Move sample_most_visited_move(const shared_ptr<MCTSNode>& node) {
	int most_visits = -1;
	Move best_move = NO_MOVE;
	for (const std::pair<Move, MCTSEdge>& p : node->outgoing_edges) {
		if (p.second.edge_visits > most_visits) {
			best_move = p.first;
			most_visits = p.second.edge_visits;
		}
	}
	assert(best_move != NO_MOVE);
	return best_move;
}

#if 0
json generate_game(int thread_id) {
	EdgeConnectState board;
//	set_board(board, STARTING_GAME_POSITION);
	MCTS mcts(thread_id, board, true);
	json entry = {{"boards", {}}, {"moves", {}}, {"dists", {}}, {"evals", {}}};
	int steps_done = 0;

#ifdef ONE_RANDOM_MOVE
	int random_ply = std::uniform_int_distribution<int>{0, 119}(generator);
	entry["random_ply"] = random_ply;
#endif

	for (unsigned int ply = 0; ply < maximum_game_plies; ply++) {
		// Do a number of steps.
		while (mcts.root_node->all_edge_visits < global_visits) {
			mcts.step();
			steps_done++;
		}
		// Sample a move according to visit counts.
		Move selected_move;
		if (ply <= 30)
			selected_move = sample_proportionally_to_visits(mcts.root_node);
		else
			selected_move = sample_most_visited_move(mcts.root_node);

#ifdef ONE_RANDOM_MOVE
		// If we're AT the randomization point then instead pick a uniformly random legal move.
		if (ply == random_ply) {
			// Pick a random move.
			int random_index = std::uniform_int_distribution<int>{
				0,
				static_cast<int>(mcts.root_node->evals.posterior.size()) - 1,
			}(generator);
			auto it = mcts.root_node->evals.posterior.begin();
			std::advance(it, random_index);
			selected_move = (*it).first;
		}

		// If we're AFTER the randomization point then pick the best move we found.
		if (ply > random_ply) {
			int most_visits = -1;
			for (const std::pair<Move, MCTSEdge>& p : mcts.root_node->outgoing_edges) {
				if (p.second.edge_visits > most_visits) {
					selected_move = p.first;
					most_visits = p.second.edge_visits;
				}
			}
		}
#endif

		// If appropriate choose a uniformly random legal move in the opening.
		if (ply < opening_randomization_schedule.size() and
		    std::uniform_real_distribution<double>{0, 1}(generator) < opening_randomization_schedule[ply]) {
			// Pick a random move.
			int random_index = std::uniform_int_distribution<int>{0, static_cast<int>(mcts.root_node->evals.posterior.size()) - 1}(generator);
			auto it = mcts.root_node->evals.posterior.begin();
			std::advance(it, random_index);
			selected_move = (*it).first;
		}

//		std::vector<char> serialized_board = serialize_board_for_json(mcts.root_board);
//		std::string s(serialized_board.begin(), serialized_board.end());
		entry["boards"].push_back(mcts.root_board.serialize_board());
		entry["moves"].push_back(std::to_string(selected_move));
		entry["evals"].push_back(mcts.root_node->get_overall_evaluation());
		entry["dists"].push_back({});
		// Write out the entire visit distribution.
		for (const std::pair<Move, MCTSEdge>& p : mcts.root_node->outgoing_edges) {
			double weight = p.second.edge_visits / mcts.root_node->all_edge_visits;
			entry["dists"].back()[std::to_string(p.first)] = weight;
		}
		mcts.play(selected_move);
		if (mcts.root_node->board.result_with_early_stopping() != 0)
			break;
	}
//	float work_factor = steps_done / (float)(global_visits * entry["moves"].size());
//	cout << "Work factor: " << work_factor << endl;

	entry["result"] = mcts.root_node->board.result_with_early_stopping();
	return entry;
}
#endif

// ================================================
//        T h r e a d e d   W o r k l o a d
// ================================================

struct Worker;
struct ResponseSlot;

std::list<Worker> global_workers;
std::vector<Worker*> global_workers_by_id;
float* global_fill_buffers[2];
int global_buffer_entries;
std::ofstream* global_output_file;

int current_buffer = 0;
int fill_levels[2] = {0, 0};
std::vector<ResponseSlot> response_slots[2];
std::queue<int> global_filled_queue;
std::mutex global_mutex;
std::atomic<bool> keep_working;

struct ResponseSlot {
	int thread_id;
};

struct Worker {
	std::mutex thread_mutex;
	std::condition_variable cv;
	std::thread t;

	bool response_filled;
	double response_value;
	float response_posterior[BOARD_SIZE * BOARD_SIZE];

	Worker(int thread_id)
		: t(Worker::thread_main, thread_id) {}

	static void thread_main(int thread_id) {
		while (true) {
			try {
			} catch (StopWorking& e) {
				return;
			}
		}
	}
};

std::pair<const float*, double> request_evaluation(int thread_id, const float* feature_string) {
	// Write an entry into the appropriate work queue.
	{
		std::lock_guard<std::mutex> global_lock(global_mutex);
		int slot_index = fill_levels[current_buffer]++;
		assert(0 <= slot_index and slot_index < response_slots[0].size());
		// Copy our features into the big buffer.
		float* destination = global_fill_buffers[current_buffer] + FEATURE_MAP_LENGTH * slot_index;
		std::copy(feature_string, feature_string + FEATURE_MAP_LENGTH, destination);
		// Place an entry requesting a reply.
		response_slots[current_buffer].at(slot_index).thread_id = thread_id;
		// Set that we're waiting on a response.
		Worker& worker = *global_workers_by_id[thread_id];
		worker.response_filled = false;
		// Swap buffers if we filled up the current one.
		if (fill_levels[current_buffer] == global_buffer_entries) {
			global_filled_queue.push(current_buffer);
			current_buffer = 1 - current_buffer;
		}
		// TODO: Notify the main thread so it doesn't have to poll.
	}
	// Wait on a reply.
	Worker& worker = *global_workers_by_id[thread_id];
	std::unique_lock<std::mutex> lk(worker.thread_mutex);
	while (not worker.response_filled) {
		worker.cv.wait_for(lk, std::chrono::milliseconds(250), [&worker]{
			return worker.response_filled;
		});
		if (not keep_working)
			throw StopWorking();
	}
	// Response collected!
	return {worker.response_posterior, worker.response_value};
}

extern "C" void launch_threads(char* output_path, int visits, float* fill_buffer1, float* fill_buffer2, int buffer_entries, int thread_count) {
	edgeconnect_initialize_structures();

	global_visits = visits;
	global_fill_buffers[0] = fill_buffer1;
	global_fill_buffers[1] = fill_buffer2;
	global_buffer_entries = buffer_entries;
	cout << "Launching into " << fill_buffer1 << ", " << fill_buffer2 << " with " << buffer_entries << " entries and " << thread_count << " threads." << endl;

	cout << "Writing to: " << output_path << endl;
	global_output_file = new std::ofstream(output_path, std::ios_base::app);
	keep_working = true;

	for (int i = 0; i < buffer_entries; i++) {
		response_slots[0].push_back(ResponseSlot());
		response_slots[1].push_back(ResponseSlot());
	}

	{
		std::lock_guard<std::mutex> global_lock(global_mutex);
		for (int i = 0; i < thread_count; i++) {
			global_workers.emplace_back(i);
			global_workers_by_id.push_back(&global_workers.back());
		}
	}
}

extern "C" int get_workload(void) {
	while (true) {
		{
			std::lock_guard<std::mutex> global_lock(global_mutex);
			// Check if a workload is ready.
			if (not global_filled_queue.empty()) {
				int workload_index = global_filled_queue.front();
				global_filled_queue.pop();
				return workload_index;
			}
		}
		std::this_thread::sleep_for(std::chrono::microseconds(100));
	}
}

extern "C" void complete_workload(int workload, float* posteriors, float* values) {
	std::lock_guard<std::mutex> global_lock(global_mutex);
	for (int i = 0; i < global_buffer_entries; i++) {
		ResponseSlot& slot = response_slots[workload].at(i);
		Worker& worker = *global_workers_by_id.at(slot.thread_id);
		worker.response_value = values[i];
		std::copy(posteriors, posteriors + BOARD_SIZE * BOARD_SIZE, worker.response_posterior);
		posteriors += BOARD_SIZE * BOARD_SIZE;
		{
			std::lock_guard<std::mutex> lk(worker.thread_mutex);
			worker.response_filled = true;
		}
		worker.cv.notify_one();
	}
	fill_levels[workload] = 0;
}

extern "C" void shutdown(void) {
	keep_working = false;
	for (Worker& w : global_workers)
		w.t.join();
	// Clear out data structures to setup for another run.
	for (int i : {0, 1})
		response_slots[i].clear();
	global_workers.clear();
	global_workers_by_id.clear();
}

