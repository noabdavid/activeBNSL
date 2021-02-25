from family import Family
from ec import EC
import numpy as np
from pgmpy.inference import VariableElimination
import itertools

class Graph:
    def __init__(self):
        self.graph = set()
        self.d = 0
        self.node_names = []
        #self.adj_mat = np.zeros((self.d, self.d))
        self.adj_mat = dict()


    def __init__(self, file_path, node_names):
        self.d = len(node_names)
        self.adj_mat = np.zeros((self.d, self.d))
        self.node_names = node_names.copy()
        f = open(file_path, 'r')
        lines = f.read().splitlines()
        self.graph = set()
        self.vars_enum = {}
        for idx, val in enumerate(self.node_names):
            self.vars_enum[val] = idx
        for i in range(self.d):
            parents = [p for p in (lines[i].split('-')[1]).split(',')[:-1]]
            self.graph.update({Family(node_names[i], parents)})  # add family
            self.adj_mat[self.vars_enum[self.node_names[i]]][self.vars_enum[parents]] = 1

        self.adj_mat.transpose()

    def __init__(self, bn, d):
        self.graph = set()
        self.d = d
        self.adj_mat = np.zeros((self.d, self.d))
        for cpd in bn.get_cpds():
            f = Family(cpd.variable, cpd.get_evidence())
            self.add_family(f)

    def add_family(self, family):
        self.graph.update({family})
        ch = family.get_child()
        par = family.get_parents()
        self.adj_mat[self.vars_enum[ch]][self.vars_enum[par]] = 1

    def __str__(self):
        return str(self.graph)

    def get_graph(self):
        return self.graph

    def getEC(self):
        edges = set()
        v_struct = set()

        for i in range(self.d):
            parents = np.where(self.adj_mat[:, i])[0]  # parents of variable i
            if len(parents) == 1:
                edges.update({tuple(sorted((i, parents[0])))})
            else:
                for pair in itertools.combinations(parents, 2):
                    p1 = pair[0]
                    p2 = pair[1]
                    if not self.adj_mat[p1][p2] and not self.adj_mat[p2][p1]:  # p1->i<-p2 is a v-structure
                        v_struct.update({(i, pair)})
                    edges.update({tuple(sorted((i, p1)))})
                    edges.update({tuple(sorted((i, p2)))})

        return EC(edges, v_struct)

    def get_adj_matrix(self):
        return self.adj_mat

    def compute_score(self, bn):
        probs = list()
        infer = VariableElimination(bn)

        # compute the local score H(f) of each family f
        for f in self.graph:
            X = f.get_child()
            Pi = list(f.get_parents())

            # if the parent set of family f is not empty, get its joint distribution
            Pi_not_empty = bool(Pi)
            if Pi_not_empty:
                Pi_states = [bn.states[p] for p in Pi]
                Pi_joint = infer.query(variables=Pi, evidence={})  # the joint distribution of the parent set

                for pi in list(itertools.product(*Pi_states)):
                    pi_as_tuples = list(zip(Pi, pi))
                    evidence = dict(pi_as_tuples)
                    q = infer.query(variables=X, evidence=evidence)  # get the distribution P[X | Pi = pi]
                    p = Pi_joint.reduce(pi_as_tuples,
                                        inplace=False)  # extract P[Pi = pi] from the joint distribution of Pi
                    probs.append(np.sum(
                        q.values * np.log2(1. / q.values)) * p.values)  # add distribution values to probability vector


            else:
                q = infer.query(variables=X, evidence={})
                probs.append(np.sum(q.values * np.log2(1. / q.values)))

        probs.sort()
        return -1 * sum(probs)  # return the score of the network



