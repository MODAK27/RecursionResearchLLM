# ── BINARY TREE DEFINITION & SAMPLING ────────────────────────────────────────
import random
from dataclasses import dataclass
from typing import Optional, List
import matplotlib.pyplot as plt
from typing import List

NUM_SAMPLES = 100

@dataclass
class TreeNode:
    val: int
    left: Optional['TreeNode'] = None
    right: Optional['TreeNode'] = None

class TreeBuilder:
    @staticmethod
    def draw_tree(nested: List, save_path: str = None):
        """
        Draws a binary tree given in nested-list format:
          [value, left_subtree_list, right_subtree_list]
        with each node as a circle containing its value, and labels the root node as 'root'.
        """
        positions = {}
        edges = []
        labels = {}
        x_counter = [0]

        def build(tree, depth=0):
            if not tree:
                return None
            val, left, right = tree
            # Build left subtree
            left_id = build(left, depth + 1)
            # Assign x based on inorder sequence
            x = x_counter[0]
            x_counter[0] += 1
            node_id = len(positions)
            positions[node_id] = (x, -depth)
            labels[node_id] = val
            # Connect edges
            if left_id is not None:
                edges.append((node_id, left_id))
            right_id = build(right, depth + 1)
            if right_id is not None:
                edges.append((node_id, right_id))
            return node_id

        # Build tree and capture root node id
        root_id = build(nested)

        plt.figure(figsize=(6, 4))
        # Draw edges
        for parent, child in edges:
            x1, y1 = positions[parent]
            x2, y2 = positions[child]
            plt.plot([x1, x2], [y1, y2], linewidth=1)
        # Draw nodes with values
        for node_id, (x, y) in positions.items():
            plt.scatter(x, y, s=300, marker='o')
            plt.text(x, y, str(labels[node_id]), ha='center', va='center')

        # Label the root node
        if root_id is not None:
            x_root, y_root = positions[root_id]
            plt.text(x_root, y_root + 0.3, 'root', ha='center', va='bottom', fontsize=10)

        plt.axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Saved tree diagram to {save_path}")
        else:
            plt.show()

    @staticmethod
    def generate_non_empty_tree(depth: int = 3, max_val: int = 100) -> TreeNode:
        """Keeps calling generate_random_tree until it returns a non‐None root."""
        tree = None
        while tree is None:
            tree = TreeBuilder.generate_random_tree(depth, max_val)
        return tree
    @staticmethod
    def generate_random_tree(depth: int = 3, max_val: int = 100) -> Optional[TreeNode]:
        """Recursively builds a random binary tree up to given depth."""
        if depth == 0 or random.random() < 0.3:
            return None
        node = TreeNode(random.randint(1, max_val))
        node.left = TreeBuilder.generate_random_tree(depth - 1, max_val)
        node.right = TreeBuilder.generate_random_tree(depth - 1, max_val)
        return node

    @staticmethod
    def tree_to_nested_list(root: Optional[TreeNode]) -> List:
        """
        Serializes tree as nested list:
          [value, left_subtree_as_list, right_subtree_as_list],
        with [] for a missing child.
        """
        if root is None:
            return []
        return [root.val,
                TreeBuilder.tree_to_nested_list(root.left),
                TreeBuilder.tree_to_nested_list(root.right)
                ]

    @staticmethod
    def inorder_traversal(root: Optional[TreeNode], out: List[int]):
        """Recursively collects inorder sequence."""
        if not root:
            return
        TreeBuilder.inorder_traversal(root.left, out)
        out.append(root.val)
        TreeBuilder.inorder_traversal(root.right, out)

    @staticmethod
    def preorder_traversal(root: Optional[TreeNode], out: List[int]):
        """Recursively collects inorder sequence."""
        if not root:
            return
        out.append(root.val)
        TreeBuilder.preorder_traversal(root.left, out)
        TreeBuilder.preorder_traversal(root.right, out)

    @staticmethod
    def generateTree(type, depth, max_val):
        tree = TreeBuilder.generate_non_empty_tree(depth=depth, max_val=max_val)
        nested = TreeBuilder.tree_to_nested_list(tree)
        ground_truth = []
        match (type):
            case "Inorder":
                TreeBuilder.inorder_traversal(tree, ground_truth)
            case "Preorder":
                TreeBuilder.preorder_traversal(tree, ground_truth)
        # Show nested representation and ground truth
        # print("Nested list representation of the tree:")
        # print(nested)
        # TreeBuilder.draw_tree(nested, save_path="Inorder/InorderTree.png")
        # print("\nGround-truth inorder traversal:")
        # print(ground_truth)
        return nested, ground_truth

    @staticmethod
    def generateDataset(type):
        nested_list = []
        groundTruth_list = []
        depth = 3
        max_val = 20
        for _ in range(NUM_SAMPLES):
            nested, groundTruth = TreeBuilder.generateTree(type, depth, max_val)
            nested_list.append(nested)
            groundTruth_list.append(groundTruth)
        return nested_list, groundTruth_list