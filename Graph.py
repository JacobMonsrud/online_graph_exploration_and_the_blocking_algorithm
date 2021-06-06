import copy
import os
import math
import heapq
import numpy
from Vertex import Vertex
from Edge import Edge
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

class Graph:

    def __init__(self, nodes: [], edges: []):
        self.adjacencyList = self.__convert_nodes_and_edges_to_adjacency_list(nodes, edges)
        self.nodes = nodes
        self.edges = edges

    def __convert_nodes_and_edges_to_adjacency_list(self, nodes: [], edges: []):
        adjacencyList = {}
        for node in nodes:
            adjacencyList[node] = []

        for edge in edges:
            adjacencyList[edge.node1].append(edge)
            adjacencyList[edge.node2].append(edge)
        return adjacencyList

    def get_known_graph(self):
        known_edges = []
        known_nodes = set()
        for edge in self.edges:
            if edge.is_boundary_edge() or edge.is_traversed():
                known_edges.append(edge)
                known_nodes.add(edge.node1)
                known_nodes.add(edge.node2)

        return Graph(list(known_nodes), known_edges)

    def draw_spacial_graph(self, title):
        for node in self.nodes:
            plt.scatter(node.point[0], node.point[1])
            plt.annotate(node.label, (node.point[0], node.point[1]))
        for edge in self.edges:
            p1, p2 = edge.line
            color = self.get_color_final(edge)[0]
            if color == 'b':
                color = 'k'
            plt.plot((p1[0], p2[0]), (p1[1], p2[1]), color)
            plt.annotate(str(edge.weight), ((p1[0] + p2[0])/2, (p1[1] + p2[1])/2))
        plt.title(title)
        plt.savefig(title)
        plt.clf()
        #plt.show()

    def label_to_node(self, label: str) -> Vertex:
        for node in self.adjacencyList.keys():
            if node.label == label:
                return node
        return None

    def initialize_single_source(self, source: Vertex):
        for node in self.adjacencyList.keys():
            node.d = numpy.inf
            node.pi = None
        source.d = 0

    def relax(self, u, v, e):
        if v.d > (u.d + e.weight):
            v.d = u.d + e.weight
            v.pi = u

    def dijkstra(self, source):
        self.initialize_single_source(source)
        S = set()
        Q = [(0, source)]
        while Q:
            u = heapq.heappop(Q)[1]
            if u in S:
                continue
            S.add(u)
            for edge in self.adjacencyList[u]:
                if edge.node1 == u:
                    v = edge.node2
                    if v.d > (u.d + edge.weight):
                        v.d = u.d + edge.weight
                        v.pi = u
                        heapq.heappush(Q, (v.d, v))
                else:
                    v = edge.node1
                    if v.d > (u.d + edge.weight):
                        v.d = u.d + edge.weight
                        v.pi = u
                        heapq.heappush(Q, (v.d, v))

    def mst_prim(self, start_node):
        edges_in_mst = set()
        for u in self.nodes:
            u.mst_key = math.inf
            u.mst_pi = None
        start_node.mst_key = 0
        Q = [(0, start_node)]
        while Q:
            u = heapq.heappop(Q)[1]  # the node is the second part of the tuple
            for edge in self.adjacencyList[u]:
                if edge.node1 == u:
                    v = edge.node2
                    if v == u.mst_pi:
                        edges_in_mst.add(edge)
                    if v.mst_key > edge.weight and edge not in edges_in_mst:
                        v.mst_pi = u
                        v.mst_key = edge.weight
                        heapq.heappush(Q, (v.mst_key, v))
                else:
                    v = edge.node1
                    if v == u.mst_pi:
                        edges_in_mst.add(edge)
                    self.is_v_in_Q(Q, v)
                    if v.mst_key > edge.weight and edge not in edges_in_mst:
                        v.mst_pi = u
                        v.mst_key = edge.weight
                        heapq.heappush(Q, (v.mst_key, v))

        return edges_in_mst

    def get_P(self):
        P = set()
        for edge in self.edges:
            if edge.inP:
                P.add(edge)
        return P

    def is_v_in_Q(self, Q, v):
        for key, node in Q:
            if node == v:
                return True
        return False

    def is_blocked(self, e, delta):
        return self.get_blocking_edge(e, delta) is not None

    def get_blocking_edge(self, e, delta):
        # blocked if
        # exists boundry edge e'

        if not e.is_boundary_edge():
            return None
        u = self.get_explored_node_of_boundary_edge(e)

        for edge in self.edges:
            # e' < e
            if edge.is_boundary_edge():
                if edge.weight < e.weight:
                    v_prime_global = self.get_unexplored_node_of_boundary_edge(edge)
                    known_graph = self.get_known_graph()
                    known_graph.dijkstra(known_graph.label_to_node(u.label))
                    v_prime_known = known_graph.label_to_node(v_prime_global.label)
                    if v_prime_known.d < (1 + delta) * e.weight:
                        return edge
                    # u->v' <= e.weight * (delta + 1)
        return None

    def get_all_blocking_edges(self, e, delta):
        # blocked if
        # exists boundry edge e'
        blocking_edges = []

        if not e.is_boundary_edge():
            return None
        u = self.get_explored_node_of_boundary_edge(e)

        for edge in self.edges:
            # e' < e
            if edge.is_boundary_edge():
                if edge.weight < e.weight:
                    v_prime_global = self.get_unexplored_node_of_boundary_edge(edge)
                    known_graph = self.get_known_graph()
                    known_graph.dijkstra(known_graph.label_to_node(u.label))
                    v_prime_known = known_graph.label_to_node(v_prime_global.label)
                    if v_prime_known.d < (1 + delta) * e.weight:
                        blocking_edges.append(edge)
                    # u->v' <= e.weight * (delta + 1)
        return blocking_edges

    # returns an unblocked boundary edge satisfying the while conditions in line 1 of the pseudocode
    def get_satisfying_edge(self, y: Vertex, delta, old_blocked_edges: list) -> Edge:
        # Look for an unblocked boundary edge starting in y.
        unblocked_boundary_starting_in_y = []  # Edges satisfying the first condition in line 1 of the pseudocode
        for edge in self.edges:
            if edge.is_boundary_edge():
                if not self.is_blocked(edge, delta):
                    u = self.get_explored_node_of_boundary_edge(edge)
                    if u == y:
                        unblocked_boundary_starting_in_y.append(edge)

        # Look for an unblocked boundary edge previously blocked by (x,y).
        prev_blocked_edges = []
        for edge in old_blocked_edges:
            if edge.is_boundary_edge():
                if not self.is_blocked(edge, delta):
                    prev_blocked_edges.append(edge)

        unblocked_boundary_starting_in_y.sort()
        prev_blocked_edges.sort()
        if unblocked_boundary_starting_in_y:
            return unblocked_boundary_starting_in_y.pop()
        elif prev_blocked_edges:
            return prev_blocked_edges.pop()
        else:
            return None

    def blocking(self, y: Vertex, delta, filename):
        old_blocked_edges = [edge for edge in self.edges if self.is_blocked(edge, delta)]

        y.explored = True
        cost = 0
        draw = False
        if filename != None:
            self.tikz(filename, delta, y)
        e = self.get_satisfying_edge(y, delta, old_blocked_edges)

        while e != None:
            if draw and filename != None:
                self.tikz(filename, delta, y)
            draw = True
            u = self.get_explored_node_of_boundary_edge(e)
            v = self.get_unexplored_node_of_boundary_edge(e)
            cost += self.walk(y, u, filename)
            cost += self.traverse(e, filename)
            cost += self.blocking(v, delta, filename)
            cost += self.walk(v, y, filename)
            e = self.get_satisfying_edge(y, delta, old_blocked_edges)

            # finished = [n.explored for n in self.nodes].all()
            # if finished:
            #     self.dijkstra(y)
            #     print("self.nodes[0].d", self.nodes[0].d)
            #     cost += self.nodes[0].d
            #     return



        return cost

    def walk(self, a: Vertex, b: Vertex, filename):
        know_graph = self.get_known_graph()
        know_graph.dijkstra(know_graph.label_to_node(a.label))
        distance = know_graph.label_to_node(b.label).d

        if filename != None:
            log = "\\\\ Walking from " + a.label + " to " + b.label + " , distance: " + str(distance) + " \\\\ \n"
            f = open(filename, "a")
            f.write(log)
            f.close()

        return distance

    def traverse(self, e: Edge, filename):
        u = self.get_explored_node_of_boundary_edge(e)
        v = self.get_unexplored_node_of_boundary_edge(e)
        e.inP = True

        if filename != None:
            log = "traversing boundary edge from " + u.label + " to " + v.label + " , weight: " + str(e.weight) + " \\\\ \n"
            f = open(filename, "a")
            f.write(log)
            f.close()

        return e.weight

    def get_unexplored_node_of_boundary_edge(self, e):
        v = e.node1 if e.node2.explored else e.node2
        return v

    def get_explored_node_of_boundary_edge(self, edge):
        u = edge.node1 if edge.node1.explored else edge.node2
        return u


    def tikz(self, filename, delta, y=None, final=False):
        f = open(filename, "a")
        f.write("\\begin{tikzpicture}\n")

        # pos of nodes
        pos = [0, 0]
        oddcounter = 0
        # Keep pos for each node, such that we can draw edges nice.
        node_pos = {}
        for node in self.nodes:
            color = "black" if node.explored else "gray"
            if node == y: color = "green"
            pos = node.point if node.point != None else [math.floor(oddcounter / 3) * 2, (oddcounter % 3) * 2]
            # pos = ([pos[0], pos[1] + 2] if oddcounter % 2 == 0 else [pos[0] + 2, pos[1]])
            f.write("\\node[shape=circle, draw=" + color + ", text=" + color + "] (" + node.label + ") at (" + str(
                pos[0]) + "," + str(pos[1]) + ") {" + node.label + "};\n")
            oddcounter += 1
            node_pos[node] = pos

        for edge in self.edges:
            if final:
                color = self.get_color_final(edge)
            else:
                color = self.get_edge_color_blocked(delta, edge)

            # need curved drawing. They are alligened on x-axis and is node between? ?
            curved = 0
            if node_pos[edge.node1][0] == node_pos[edge.node2][0] and node_pos[edge.node1][1] + node_pos[edge.node2][
                1] == 4:
                curved = 45
            # Are they on the same row, but not right next to eachother
            if node_pos[edge.node1][1] == node_pos[edge.node2][1] and abs(
                    node_pos[edge.node1][0] - node_pos[edge.node2][0]) > 2:
                # top or bottom?
                if node_pos[edge.node1][1] == 4:
                    curved = 45
                else:
                    curved = -45
            # Not on same x-axis
            if node_pos[edge.node1][0] != node_pos[edge.node2][0] and node_pos[edge.node1][1] != node_pos[edge.node2][
                1]:
                curved = 15
            curved = 0 if node.point != None else curved
            curved = 0
            # Should weight be above or to the left of the edge
            pos_weight = "above"
            if node_pos[edge.node1][0] == node_pos[edge.node2][0]:
                pos_weight = "left"

            # Find what edge is blocking
            blocked_by = ""
            if self.get_all_blocking_edges(edge, delta) != None:
                for ed in self.get_all_blocking_edges(edge, delta):
                    n1 = self.get_explored_node_of_boundary_edge(ed)
                    n2 = self.get_unexplored_node_of_boundary_edge(ed)
                    blocked_by += " (" + str(n1.label) + ", " + str(n2.label) + ") "

            f.write("\path (" + edge.node1.label + ") [draw=" + color + ", bend left=" + str(
                curved) + "] edge node[" + pos_weight + ", text=" + color + "] {$" + str(
                edge.weight) + "$" + blocked_by + "} (" + edge.node2.label + ");\n")

        f.write("\end{tikzpicture}\n")
        f.close()

    def get_edge_color_blocked(self, delta, edge):
        color = "gray"
        if edge.is_boundary_edge():
            color = "blue"
            if self.is_blocked(edge, delta):
                color = "red"
        elif edge.is_traversed():
            color = "black"
        return color

    def get_color_final(self, edge):
        color = "black"
        if edge.inP:
            color = "green"
            if edge in self.mst_prim(edge.node1):
                color = "cyan"
        elif edge in self.mst_prim(edge.node1):
            color = "yellow"
        return color

    def toString(self):
        s = "Nodes: "
        for node in self.nodes:
            s += node.label + " "
        s += "\nEdges: "
        for edge in self.edges:
            s += edge.toString() + " "
        return s

    def mst_scipy(self):
        n = len(self.nodes)
        matrix = np.zeros((n, n))
        node_indexes = {}
        index = 0
        for node in self.nodes:
            node_indexes[node] = index
            index += 1

        for edge in self.edges:
            index1 = node_indexes[edge.node1]
            index2 = node_indexes[edge.node2]
            matrix[index1][index2] = edge.weight
            matrix[index2][index1] = edge.weight
        mst = minimum_spanning_tree(matrix)
        print(mst)

