class Vertex:
    def __init__(self, label: str):
        self.label = label
        self.d = None
        self.pi = None
        self.explored = False
        self.mst_key = None
        self.mst_pi = None
        self.point = None

    def __str__(self):
        return self.label

    def set_point(self, point):
        self.point = point

    #def __eq__(self, other):
    #    return self.label == other.label and self.d == other.d and self.pi == other.pi and self.explored == other.explored

    def __gt__(self, other):
        if self.d == None or other.d == None:
            return self.label > other.label
        return self.d > other.d