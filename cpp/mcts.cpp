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
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/graph/default_device.h>

#include "json.hpp"
#include "edgeconnect_rules.h"

using std::shared_ptr;
using std::cout;
using std::endl;

extern "C" double exploration_parameter = 1.0;
constexpr double dirichlet_alpha = 0.03;
constexpr double dirichlet_weight = 0.25;
constexpr int thread_count = 2;

std::random_device rd;
std::default_random_engine generator(rd());
//std::default_random_engine generator(123456789);

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

constexpr const char* meta_graph_path = "cpp/checkpoints/edgeconnect-model.meta";
constexpr const char* checkpoint_path = "cpp/checkpoints/edgeconnect-model";

static inline int get_random_symmetry() {
//	return 0;
	return std::uniform_int_distribution<int>{0, 11}(generator);
}

std::vector<int> ensemble_sizes;
int cache_hits;
uint64_t mcts_search_hash;

static inline uint64_t combine_nonlinear(uint64_t x, uint64_t y) {
	x ^= (x << 5) + y + (x >> 2);
	x *= 7;
	return x;
}

void snp_copy(float* start, float* stop, float* dest) {
	assert(stop - start == BOARD_SIZE * BOARD_SIZE);
	std::copy(start, stop, dest);
}

struct BoardEvaluator {
	static constexpr int MAX_ENSEMBLE_SIZE = 4;
	static constexpr int QUEUE_DEPTH = 4096;
	static constexpr float PROBABILITY_THRESHOLD = 0.1; //005;
	static constexpr int MAX_CACHE_ENTRIES = 10000;

	struct NNResult {
		float policy[BOARD_SIZE * BOARD_SIZE];
		double value;
		int random_symmetry;
	};

	struct BoardQueue {
		std::mutex queue_mutex;
		std::deque<EdgeConnectState> board_queue;
	};

	std::mutex cache_mutex;
	std::unordered_map<size_t, std::unique_ptr<NNResult>> nn_cache;
	std::vector<BoardQueue> per_thread_board_queues;
	std::deque<EdgeConnectState> board_queue;

	tensorflow::Session* session;
	tensorflow::MetaGraphDef meta_graph_def;
	tensorflow::Tensor input_tensor;

	BoardEvaluator(int thread_count)
		: per_thread_board_queues(thread_count)
		, input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({MAX_ENSEMBLE_SIZE, 23, 23, 12}))
	{
		auto opts = tensorflow::SessionOptions();
		opts.config.mutable_gpu_options()->set_allow_growth(true);
		std::cout << "Devices: " << opts.config.gpu_options().visible_device_list() << std::endl;
		session = tensorflow::NewSession(opts);
		if (session == nullptr)
			throw std::runtime_error("Could not create Tensorflow session.");
		tensorflow::Status status;

		tensorflow::MetaGraphDef meta_graph_def;
		status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), meta_graph_path, &meta_graph_def);
		if (!status.ok())
			throw std::runtime_error("Error reading meta-graph definition from " + std::string(meta_graph_path) + ": " + status.ToString());
		auto graph_def = meta_graph_def.graph_def();

		status = session->Create(graph_def);
		if (!status.ok())
			throw std::runtime_error("Error creating graph: " + status.ToString());

		// Read weights from the saved checkpoint
		tensorflow::Tensor checkpointPathTensor(tensorflow::DT_STRING, tensorflow::TensorShape());
		checkpointPathTensor.scalar<std::string>()() = checkpoint_path;
		status = session->Run(
			{{meta_graph_def.saver_def().filename_tensor_name(), checkpointPathTensor}},
			{},
			{meta_graph_def.saver_def().restore_op_name()},
			nullptr
		);
		if (!status.ok())
			throw std::runtime_error("Error loading checkpoint from " + std::string(checkpoint_path) + ": " + status.ToString());
		std::cout << "Loaded." << std::endl;

		auto eigen_tensor = input_tensor.flat<float>();
		for (int i = 0; i < MAX_ENSEMBLE_SIZE * 23 * 23 * 12; i++)
			eigen_tensor(i) = 0.0;
	}

	void do_practice_computation() {
		std::vector<tensorflow::Tensor> outputTensors;
		tensorflow::Tensor is_training(tensorflow::DT_BOOL, tensorflow::TensorShape());
		is_training.scalar<bool>()() = false;
		tensorflow::Status status = session->Run(
			{{"net/input_placeholder", input_tensor}, {"net/is_training", is_training}},
			{"net/policy_output_id", "net/value_output_id"},
			{},
			&outputTensors
		);
		if (!status.ok())
			throw std::runtime_error("Error evaluating: " + status.ToString());
	}

	int compute_evaluation(int thread_id, const EdgeConnectState& board, float* policy, double* value) {
		size_t board_hash = std::hash<EdgeConnectState>{}(board);
		// Unqueue some boards to be computed along with this one.
		{
			std::unique_lock<std::mutex> lk(cache_mutex);
			auto it = nn_cache.find(board_hash);
			if (it != nn_cache.end()) {
//				std::cout << " --------------- Cache hit!" << std::endl;
				cache_hits++;
				NNResult& cache_entry = *it->second;

#if 0
				{
					board.featurize(0, &input_tensor.flat<float>()(0));
					std::vector<tensorflow::Tensor> output_tensors;
					tensorflow::Tensor is_training(tensorflow::DT_BOOL, tensorflow::TensorShape());
					is_training.scalar<bool>()() = false;
					tensorflow::Status status = session->Run(
						{{"net/input_placeholder", input_tensor}, {"net/is_training", is_training}},
						{"net/policy_output_id", "net/value_output_id"},
						{},
						&output_tensors
					);
					if (!status.ok())
						throw std::runtime_error("Error evaluating: " + status.ToString());
					// Make sure our result is the same.
					float reference_value = output_tensors[1].flat<float>()(0);
					if (fabs(reference_value - cache_entry.value) > 1e-7) {
						std::cout << " ~~~~~~~~~~~ Divergent values: " << cache_entry.value << " should have been " << reference_value << std::endl;
					}
				}
#endif

				snp_copy(&cache_entry.policy[0], &cache_entry.policy[BOARD_SIZE * BOARD_SIZE], policy);
				*value = cache_entry.value;
				return cache_entry.random_symmetry;
			}
		}

		int random_symmetry = get_random_symmetry(); //std::uniform_int_distribution<int>{0, 11}(generator);
		board.featurize(random_symmetry, &input_tensor.flat<float>()(0));
		int loaded_up = 1;

		std::vector<NNResult*> cache_entries;
		while (loaded_up < MAX_ENSEMBLE_SIZE and board_queue.size() > 0) {
			const EdgeConnectState& queued_board = board_queue.front();
			auto [it, newly_inserted] = nn_cache.emplace(
				std::hash<EdgeConnectState>{}(queued_board),
				std::make_unique<NNResult>()
			);
			if (not newly_inserted) {
//				std::cout << "This should not happen!!!!!!!!!!!!!!!!!!" << std::endl;
				board_queue.pop_front();
				continue;
			}
			/*
			auto it = nn_cache.find(board_hash);
			if (it != nn_cache.end()) {
				board_queue.pop_front();
				continue;
			}
			*/
//			NNResult* cache_entry_ptr = &nn_cache[state];
			NNResult* cache_entry_ptr = it->second.get();
			cache_entries.push_back(cache_entry_ptr);
			cache_entry_ptr->random_symmetry = get_random_symmetry(); //std::uniform_int_distribution<int>{0, 11}(generator);
			int float_offset = FEATURE_MAP_LENGTH * loaded_up;
			queued_board.featurize(cache_entry_ptr->random_symmetry, &input_tensor.flat<float>()(float_offset));
			board_queue.pop_front();
			loaded_up++;
		}

//		std::cout << "Running on: " << loaded_up << std::endl;
		ensemble_sizes.push_back(loaded_up);

		std::vector<tensorflow::Tensor> output_tensors;
		tensorflow::Tensor is_training(tensorflow::DT_BOOL, tensorflow::TensorShape());
		is_training.scalar<bool>()() = false;
		tensorflow::Status status = session->Run(
			{{"net/input_placeholder", input_tensor}, {"net/is_training", is_training}},
			{"net/policy_output_id", "net/value_output_id"},
			{},
			&output_tensors
		);
		if (!status.ok())
			throw std::runtime_error("Error evaluating: " + status.ToString());

		auto output_policy = output_tensors[0].flat<float>();
		snp_copy(&output_policy(0), &output_policy(BOARD_SIZE * BOARD_SIZE), policy);
		*value = output_tensors[1].flat<float>()(0);

		for (int i = 1; i < loaded_up; i++) {
			NNResult& cache_entry = *cache_entries.at(i - 1);
			int float_offset = BOARD_SIZE * BOARD_SIZE * i;
			snp_copy(&output_policy(float_offset), &output_policy(float_offset + BOARD_SIZE * BOARD_SIZE), cache_entry.policy);
			cache_entry.value = output_tensors[1].flat<float>()(i);
		}

		return random_symmetry;
	}

	bool can_accept_more() {
		return board_queue.size() < QUEUE_DEPTH;
	}

	bool add_likely_to_be_needed_board(int thread_id, const EdgeConnectState& board, Move move) {
		if (not can_accept_more())
			return false;
		if (nn_cache.find(std::hash<EdgeConnectState>{}(board)) != nn_cache.end())
			return false;
//		board_queue.emplace_back(board);
		board_queue.emplace_back(board);
		board_queue.back().make_move(move);
		return true;
	}
};

std::unique_ptr<BoardEvaluator> global_evaluator;

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

//		int random_symmetry = std::uniform_int_distribution<int>{0, 11}(generator);
//		float feature_buffer[FEATURE_MAP_LENGTH] = {};
//		board.featurize(random_symmetry, feature_buffer);
		float posterior_array[BOARD_SIZE * BOARD_SIZE];
		int random_symmetry = global_evaluator->compute_evaluation(thread_id, board, posterior_array, &value);
//		std::cout << "Got: " << random_symmetry << std::endl;

		// Softmax the posterior array.
		double softmaxed[BOARD_SIZE * BOARD_SIZE];
		for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++)
			softmaxed[i] = exp(posterior_array[MOVE_SYMMETRY_LOOKUP[random_symmetry][i]]);
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

		// Find children that are likely to be needed.
		if (global_evaluator->can_accept_more())
			for (auto& p : posterior)
				if (p.second >= BoardEvaluator::PROBABILITY_THRESHOLD)
					if (global_evaluator->add_likely_to_be_needed_board(thread_id, board, p.first))
						1;
//						std::cout << "Added entry for: " << p.first << " with " << p.second << std::endl;

		// Add Dirichlet noise.
		if (use_dirichlet_noise) {
			assert(false);
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
			const MCTSEdge& edge = it->second;
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
			if (not (score >= 0.0))
				std::cout << "Bad score: " << score << std::endl;
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
		std::weak_ptr<MCTSNode>& cache_entry = it->second;
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
				move = it->first;
			} else {
				// Pick the edge that has the current highest k-armed bandit value.
				move = node->select_action();
			}
			// If the tree doesn't continue in the direction of this move, then break.
			const auto it = node->outgoing_edges.find(move);
			if (it == node->outgoing_edges.end())
				break;
			MCTSEdge& edge = it->second;
			edges_on_path.push_back(&edge);
			node = edge.child_node;
		}
		return std::tuple<shared_ptr<MCTSNode>, Move, std::vector<MCTSEdge*>>{node, move, edges_on_path};
	}

	void step() {
		// 1) Pick a path through the tree.
		auto [leaf_node, move, edges_on_path] = select_principal_variation();
//		// Darn, I wish I had structured bindings already. :(
//		shared_ptr<MCTSNode>         leaf_node     = std::get<0>(triple);
//		Move                         move          = std::get<1>(triple);
//		std::vector<MCTSEdge*>       edges_on_path = std::get<2>(triple);

		shared_ptr<MCTSNode> new_node;

//		std::cout << "Visit:";
		for (auto e : edges_on_path) {
//			std::cout << " " << e->edge_move;
			mcts_search_hash = combine_nonlinear(mcts_search_hash, e->edge_move);
		}
		mcts_search_hash = combine_nonlinear(mcts_search_hash, 12345);
//		std::cout << std::endl;

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
			MCTSEdge& new_edge = pair_it_success.first->second;
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
		root_node = it->second.child_node;
		root_board = root_node->board;
		// Now that a new node is the root we have to redo its evals with Dirichlet noise, if required.
		// This is a little wasteful, when we could just apply Dirichlet noise, but it's not that bad.
		root_node->evals_populated = false;
		root_node->populate_evals(thread_id, use_dirichlet_noise);
	}

	void set_state_reusing_subtree_if_possible(const EdgeConnectState& board) {
		shared_ptr<MCTSNode> new_root = nullptr;
		// Search at depth 1 for the board.
		for (auto& edge : root_node->outgoing_edges) {
			if (edge.second.child_node->board == board) {
				new_root = edge.second.child_node;
				break;
			}
			// Search at depth 2 for the board.
			for (auto& edge2 : edge.second.child_node->outgoing_edges) {
				if (edge2.second.child_node->board == board) {
					new_root = edge2.second.child_node;
					break;
				}
			}
		}
		if (new_root != nullptr) {
			std::cerr << "Managed to reuse some subtree." << std::endl;
			root_node = new_root;
			root_board = root_node->board;
			return;
		}
		// Oh no, we couldn't reuse a subtree.
		init_from_scratch(board);
	}
};

Move sample_proportionally_to_visits(const shared_ptr<MCTSNode>& node) {
//	assert(false);
	double x = std::uniform_real_distribution<float>{0, 1}(generator);
	for (const std::pair<Move, MCTSEdge>& p : node->outgoing_edges) {
		double weight = p.second.edge_visits / node->all_edge_visits;
		if (x <= weight)
			return p.first;
		x -= weight;
	}
	// If we somehow slipped through then return some arbitrary element.
	std::cerr << "Potential bug: Weird numerical edge case in sampling!" << endl;
	return node->outgoing_edges.begin()->first;
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

std::unique_ptr<MCTS> global_mcts;

extern "C" void initialize() {
	edgeconnect_initialize_structures();
	global_evaluator = std::make_unique<BoardEvaluator>(2);
	global_mcts = std::make_unique<MCTS>(0, EdgeConnectState{}, false);
}

extern "C" void set_state_from_string(char* board_string) {
//	std::cout << "I was called!" << std::endl;
	EdgeConnectState board;
	board.from_string(board_string);
	// FIXME: Keep subtree progress here!
	// This will be especially important once I implement pondering.
	global_mcts->set_state_reusing_subtree_if_possible(board);
//	global_mcts = std::make_unique<MCTS>(0, board, false);
}

extern "C" int get_visits_in_current_tree() {
	int total = 0;
	for (auto& e : global_mcts->root_node->outgoing_edges)
		total += e.second.edge_visits;
	return total;
}

extern "C" int think() {
	global_mcts->step();
	// Compute the best move so far.
	return sample_proportionally_to_visits(global_mcts->root_node);
//	return sample_most_visited_move(global_mcts->root_node);
}

extern "C" void launch_mcts(int step_count) {
	ensemble_sizes.clear();
	cache_hits = 0;
	mcts_search_hash = 123456789;

//	global_evaluator->nn_cache.clear();
//	global_evaluator->board_queue.clear();

	EdgeConnectState starting_state;
	MCTS mcts(0, starting_state, false);
	for (int i = 0; i < step_count; i++)
		mcts.step();

	double mean_ensemble_size = 0;
	int min_ensemble_size = 1000;
	int max_ensemble_size = -1000;
	for (auto i : ensemble_sizes) {
		mean_ensemble_size += i;
		min_ensemble_size = std::min(min_ensemble_size, i);
		max_ensemble_size = std::max(max_ensemble_size, i);
	}
	std::cout << "Mean ensemble size: " << mean_ensemble_size / ensemble_sizes.size() << " Range: " << min_ensemble_size << " - " << max_ensemble_size << std::endl;
	std::cout << "Cache hits: " << cache_hits << std::endl;
	std::cout << "Search hash: " << mcts_search_hash << std::endl;
}

