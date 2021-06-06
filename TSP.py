from Graph import Graph
from Edge import Edge
from Vertex import Vertex
import itertools
import math
import copy
import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming as tsp_solver

class TSP:

    def __init__(self):
        pass

    def get_edge_cost(self, prev_node: Vertex, node: Vertex, graph: Graph):
        for edge in graph.adjacencyList[prev_node]:
            if edge.node1 == node or edge.node2 == node:
                return edge.weight
        return math.inf

    def python_tspm_solver(self, graph: Graph):
        tspm_grap = self.convert_tspm_to_tsp_graph(graph)
        n = len(tspm_grap.nodes)
        matrix = np.zeros((n, n))
        node_indexes = {}
        index = 0
        for node in tspm_grap.nodes:
            node_indexes[node] = index
            index += 1

        for edge in tspm_grap.edges:
            index1 = node_indexes[edge.node1]
            index2 = node_indexes[edge.node2]
            matrix[index1][index2] = edge.weight
            matrix[index2][index1] = edge.weight

        perm, cost = tsp_solver(matrix)
        return cost, perm

    def TSP(self, graph: Graph):
        # Brute force. Try all permutaions
        # We can asume the first node will be start and end.
        start_and_end_node = graph.nodes[0]
        rest_nodes = graph.nodes[1:]
        permutations_of_nodes = itertools.permutations(rest_nodes)
        min_cost = math.inf
        min_path = ""
        for permutation in permutations_of_nodes:
            current_path = 0
            # Find path in permutation
            prev_node = start_and_end_node

            for node in permutation:
                current_path += self.get_edge_cost(prev_node, node, graph)
                prev_node = node
            current_path += self.get_edge_cost(prev_node, start_and_end_node, graph)

            if current_path < min_cost:
                min_cost = current_path
                s = start_and_end_node.label + " -> "
                for node in permutation:
                    s += node.label + " -> "
                min_path = s
        return min_cost, min_path

    # TSP multiple visits, not fully connected
    def TSPM(self, graph: Graph):
        tsp_grap = self.convert_tspm_to_tsp_graph(graph)
        tsp_cost, min_path = self.TSP(tsp_grap)
        return tsp_cost, min_path


    def convert_tspm_to_tsp_graph(self, graph: Graph) -> Graph:
        # all pair shortest path
        # new graph (fully connected) where edges have weight as shortest path between nodes
        tsp_nodes = []
        tsp_edges = []
        labels_done_nodes = []
        for node in graph.nodes:
            tsp_nodes.append(node)
            graph.dijkstra(node)
            for node_inner in graph.nodes:
                if node_inner.label != node.label and node_inner.label not in labels_done_nodes:
                    weight = graph.label_to_node(node_inner.label).d
                    tsp_edges.append(Edge(node, node_inner, weight))
            labels_done_nodes.append(node.label)

        return Graph(tsp_nodes, tsp_edges)
