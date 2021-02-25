class EC:

    def __init__(self, edges, vstruct):
        self.edges = edges
        self.vstruct = vstruct

    def get_edges(self):
        return self.edges

    def get_vstruct(self):
        return self.vstruct

    def compare(self, e2):
        return self.edges == e2.edges and self.vstruct == e2.vstruct