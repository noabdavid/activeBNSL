import math
import numpy as np
from heapq import nlargest


def sample_size(epsilon, delta, Ma, Mb):
    x = 2 * math.log(2 / delta) / (epsilon ** 2)
    N = math.ceil(x * (math.log(x) ** 2))
    return max([N, Ma, 2 * (Ma - 1) * Mb / epsilon, math.exp(1) ** 2])

def support(states, k):
""" returns the maximal cardinality of a variable in data (Ma) and the maximal cardinality of a parent set of size k (Mb)
 Parameters
        ----------
        states  : dict
                  states[i] is a list with node i support
        k       : int
                  maximal in-degree
"""
    unique_counts = [len(l) for l in states]
    Ma = max(unique_counts)
    k_largest = nlargest(k, unique_counts)
    Mb = np.prod(k_largest)

    return Ma, Mb
