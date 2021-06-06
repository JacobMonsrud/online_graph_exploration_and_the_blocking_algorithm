from Vertex import Vertex

class Edge:
    def __init__(self, node1: Vertex, node2: Vertex, weight):
        self.weight = weight
        self.node1 = node1
        self.node2 = node2
        self.inP = False  # P is the set of edges traversed in line 3 of the pseudo-code
        self.line = None

    #def __eq__(self, other):
    #    return self.weight == other.weight and self.node1 == other.node1 and self.node2 == other.node2

    def set_line(self, line):
        self.line = line

    def toString(self):
        return self.node1.label + "-" + self.node2.label + ":" + str(self.weight)

    def __gt__(self, other):
        return self.node1.label + self.node2.label < other.node1.label + other.node2.label

    def is_boundary_edge(self):
        return self.node1.explored != self.node2.explored

    def is_traversed(self):
        return self.node1.explored and self.node2.explored
