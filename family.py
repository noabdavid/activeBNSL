class Family:
    def __init__(self, child):
        self.child = child
        self.parents = tuple()

    def __init__(self, child, parents):
        self.child = child
        self.parents = tuple(parents)

    def get_child(self):
        return self.child

    def get_parents(self):
        return self.parents

    def num_families(self):
        return len(self.parents)