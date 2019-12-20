
#ifndef UNION_FIND_H
#define UNION_FIND_H

#include <cassert>
#include <vector>
#include <unordered_map>

template <typename Key>
struct UnionFind {
	typedef size_t NodeIndex;

	struct UnionFindNode {
		NodeIndex parent;
		int rank;
		Key key;
	};

	std::vector<UnionFindNode> nodes;
	std::unordered_map<Key, NodeIndex> key_to_node_index;

	NodeIndex make_node(Key key) {
		auto it = key_to_node_index.find(key);
		assert(it == key_to_node_index.end());
		NodeIndex new_node_index = nodes.size();
		nodes.push_back({
			new_node_index, // node.parent points to itself.
			0, // node.rank = 0
			key // node.key = key
		});
		// Add the appropriate mapping.
		key_to_node_index[key] = new_node_index;
		return new_node_index;
	}

	inline bool contains(Key key) {
		return key_to_node_index.count(key) != 0;
	}

	inline NodeIndex find(Key key) {
		// Make the node if it doesn't exist yet.
		if (key_to_node_index.count(key) == 0)
			make_node(key);
		return find_by_index(key_to_node_index.at(key));
	}

	inline NodeIndex find_by_index(NodeIndex index) {
		UnionFindNode& node = nodes[index];
		if (node.parent != index) {
			NodeIndex new_parent = find_by_index(node.parent);
			node.parent = new_parent;
		}
		return node.parent;
	}

	inline void union_nodes(Key k1, Key k2) {
		// FIXME: DRY this with above and refactor. This is ugly. :(
		if (key_to_node_index.count(k1) == 0)
			make_node(k1);
		if (key_to_node_index.count(k2) == 0)
			make_node(k2);
		union_nodes_by_index(key_to_node_index.at(k1), key_to_node_index.at(k2));
	}

	inline void union_nodes_by_index(NodeIndex x, NodeIndex y) {
		NodeIndex x_root_index = find_by_index(x), y_root_index = find_by_index(y);
		if (x_root_index == y_root_index)
			return;
		UnionFindNode& x_root = nodes[x_root_index];
		UnionFindNode& y_root = nodes[y_root_index];
		if (x_root.rank < y_root.rank) {
			x_root.parent = y_root_index;
		} else {
			y_root.parent = x_root_index;
			// Increment x_root's rank to indicate that y_root was just rerooted to it.
			if (x_root.rank == y_root.rank)
				x_root.rank++;
		}
	}

	inline bool check_same_set(Key k1, Key k2) {
		return find(k1) == find(k2);
	}

	inline UnionFindNode& get_by_index(NodeIndex x) {
		return nodes[x];
	}

	inline const UnionFindNode& get_by_index(NodeIndex x) const {
		return nodes[x];
	}
};

#endif

