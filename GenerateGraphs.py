import copy

from Graph import Graph
from Edge import Edge
from Vertex import Vertex
import random
from enum import Enum

class GenarateGraphs:

    def __init__(self):
        pass

    def generate(self, number_of_graphs: int, size: int, connectivity: int, weigths_set=None):
        graphs = []
        for i in range(number_of_graphs):
            weigths_set_copy = copy.copy(weigths_set)
            graph = self.generate_graph(size, connectivity, weigths_set_copy)
            graphs.append(graph)
        return graphs

    def generate_graph(self, size: int, connectivity: int, weight_set=None):
        label_counter = 0
        nodes_pos = {}
        nodes = []
        edges = []

        for i in range(size):
            node = Vertex(str(label_counter))
            nodes.append(node)
            coordinates = self.__generate_coordinates(size)
            while coordinates in nodes_pos.values():
                coordinates = self.__generate_coordinates(size)
            nodes_pos[node] = coordinates
            node.set_point(coordinates)
            label_counter += 1

        # Make sure they are connected
        conected = []
        conected.append(nodes[0])
        lines = []
        for node in nodes[1:]:
            nodes_tried = set()
            node2 = random.choice(conected)
            nodes_tried.add(node2)
            line = (nodes_pos[node], nodes_pos[node2])
            while self.line_intersect_list_of_lines(line, lines) or self.intersect_any_node(line, nodes, nodes_pos, node, node2):

                # There is no solution, we will loop. Start over
                if len(nodes_tried) == len(conected):
                    return self.generate_graph(size, connectivity)

                node2 = random.choice(conected)
                nodes_tried.add(node2)
                line = (nodes_pos[node], nodes_pos[node2])
            lines.append(line)
            conected.append(node)
            weight = weight_set.pop(0) if weight_set is not None else random.randint(1, 10000)
            e = Edge(node, node2, weight)
            e.set_line(line)
            edges.append(e)

        # Genereate some random edges according to connectivity
        for node in nodes:
            for node_inner in nodes:
                if node != node_inner:
                    if connectivity > random.randint(0, 100):
                        line_edge = (nodes_pos[node], nodes_pos[node_inner])
                        if not self.line_intersect_list_of_lines(line_edge, lines):
                            if not self.in_edges(node, node_inner, edges):
                                lines.append(line_edge)
                                weight = weight_set.pop(0) if weight_set is not None else random.randint(1, 10000)
                                e = Edge(node, node_inner, weight)
                                e.set_line(line_edge)
                                edges.append(e)
        g = Graph(nodes, edges)
        return g

    def in_edges(self, node1, node2, edges):
        for edge in edges:
            if node1 in [edge.node1, edge.node2] and node2 in [edge.node1, edge.node2]:
                return True
        return False

    def intersect_any_node(self, line, nodes, nodes_pos, node1, node2):
        p1, p2 = line
        for node in nodes:
            if node != node1 and node != node2:
                n1 = nodes_pos[node]
                if Orientation.COLLINEAR == self.orientation(p1, p2, n1):
                    # does n1 lie between p1 and p2? if so, return true!!
                    x1, y1 = p1
                    x2, y2 = p2
                    n1, n2 = n1
                    if n1 >= min(x1, x2) and n1 <= max(x1, x2):
                        return True
        return False


    def __generate_coordinates(self, size: int):
        pos_x = random.randint(0, 2 * size)
        pos_y = random.randint(0, 2 * size)
        return pos_x, pos_y

    def line_intersect_list_of_lines(self, line1, line_list):
        for line in line_list:
            if self.lines_intersect(line1, line):
                return True
        return False

    def lines_intersect(self, line1, line2):
        # Sepcial case when they start in the same point
        p1, p2 = line1
        q1, q2 = line2
        if p1 in [q1, q2]:
            return Orientation.COLLINEAR == self.orientation(p2, q1, q2) and self.parrellel_lines_intersect(line1, line2)
        elif p2 in [q1, q2]:
            return Orientation.COLLINEAR == self.orientation(p1, q1, q2) and self.parrellel_lines_intersect(line1, line2)

        if self.orientation(p1, p2, q1) != self.orientation(p1, p2, q2) and self.orientation(q1, q2, p1) != self.orientation(q1, q2, p2):
            return True
        elif self.orientation(p1, p2, q1) == self.orientation(p1, p2, q2) == self.orientation(q1, q2, p1) == self.orientation(q1, q2, p2) == Orientation.COLLINEAR:
            return self.parrellel_lines_intersect(line1, line2)

        return False


    def parrellel_lines_intersect(self, line1, line2):
        x1 = line1[0][0]
        y1 = line1[0][1]
        x2 = line1[1][0]
        y2 = line1[1][1]

        x3 = line2[0][0]
        y3 = line2[0][1]
        x4 = line2[1][0]
        y4 = line2[1][1]
        return not (min(x1, x2) >= max(x3, x4) or max(x1, x2) <= min(x3, x4))

        #return ((x1 < x3 and x1 < x4) == (x2 < x3 and x2 < x4)) and ((y1 < y3 and y1 < y4) == (y2 < y3 and y2 < y4))

    def orientation(self, p1, p2, p3):
        res = (p2[1] - p1[1]) * (p3[0] - p2[0]) - (p3[1] - p2[1]) * (p2[0] - p1[0])
        if res == 0:
            return Orientation.COLLINEAR
        elif res > 0:
            return Orientation.CLOCKWISE
        else:
            return Orientation.COUNTERCLOCKWISE

class Orientation(Enum):
    CLOCKWISE = 1
    COUNTERCLOCKWISE = 2
    COLLINEAR = 3
