from pgmpy.readwrite import BIFReader
from pgmpy.inference import VariableElimination
import uuid
import math
from heapq import nlargest
import numpy as np
import panda as pd
import itertools
import os
from family import Family

def load_network(net_name):
    reader = BIFReader('./BIF/' + net_name + '.bif')
    states = reader.get_states()  # get all node states
    node_names = list(states.keys())  # get node names
    model = reader.get_model()
    return model, node_names, states

def generate_id():
    return str(uuid.uuid1())

def N(epsilon, delta, Ma, Mb):
    x = 8 * math.log(2 / delta) / (epsilon ** 2)
    N = math.ceil(x * (math.log(x) ** 2))
    return max([N, Ma, 2 * (Ma - 1) * Mb / epsilon, math.exp(1) ** 2])

def support(states, k):
    unique_counts = [len(l) for l in states]
    Ma = max(unique_counts)
    k_largest = nlargest(k, unique_counts)
    Mb = np.prod(k_largest)
    return Ma, Mb

def sample_subsets(k, model, states, N, active_vars=[]):
    infer = VariableElimination(model)  # query object
    node_names = list(model.nodes())
    sample = pd.DataFrame(columns=node_names + ['count'])

    for S in itertools.combinations(node_names, k + 1):
        # given a set of active nodes, sample only subsets in which at least one variable is active
        if not active_vars or bool(set(S).intersection(set(active_vars))):
            q = infer.query(variables=list(S), evidence={})  # joint distribution of variables
            counts = np.random.multinomial(N, (q.values).flatten())  # sample of size N from q
            curr_states = [states[s] for s in S]
            curr_sample = pd.DataFrame(list(itertools.product(*curr_states)), columns=q.scope())
            curr_sample['count'] = counts
            sample = sample.append(curr_sample).fillna(np.nan)
    return sample

def update_score_file(d, vars, k, counts, exists_file, file_path):
    # if file exists copy its content for editing, else create a new file
    if exists_file:
        f = open(file_path, "r")
        content = f.read().splitlines()
    else:
        f = open(file_path, "w+")
        content = []
        content.append(str(d))  #  first line of the file is the number of variables

    i = 1  # position in file
    dict = {}
    candidates = vars

    for X in vars:
        dict[X] = set()  # keep track after families that their score was already computed
        print('Computing scores for variable ' + X)
        if exists_file:
            # go to X's section in the file
            lenX = len(X)
            while X != content[i][0:lenX]:
                print('searching for variable ' + X + ' in ' + str(content[i][0]))
                i = i + 1
        else:
            num_parent_sets = 0

        candidates_tag = [c for c in candidates if c != X]
        for k_tag in range(k+1):
            for Pi in itertools.combinations(candidates_tag, k_tag):
                print('Parent set: ' + str(Pi))
                Pi_tag = frozenset(Pi)
                if Pi_tag not in dict[X]:
                    dict[X].add(Pi_tag)
                    # compute score
                    family_score = compute_score(X, Pi, counts)
                    # go to the line where the score should be updated
                    if exists_file:

                        if not Pi:
                            st = [str(k_tag)] + ['']
                        else:
                            st = [str(k_tag)] + [x for x in Pi]
                        while st != content[i].split(' ')[1:]:
                            i = i + 1
                        # rewrite line with updated score
                        content[i] = str(family_score) + ' ' + str(k_tag) + ' ' + ' '.join(Pi)

                    else:
                        num_parent_sets += 1
                        content.insert(i + num_parent_sets, str(family_score) + ' ' + str(k_tag) + ' ' + ' '.join(Pi))
        # if the file is first created add the number of parent sets per variable
        if not exists_file:
            content.insert(i, X + ' ' + str(num_parent_sets))
            i = i + num_parent_sets + 1

    if exists_file:
        f = open(file_path, 'w')

    f.write('\n'.join(content))

def compute_score(X, Pi, counts):
    # get counts of joint distribution of X, Pi
    joint_counts = counts[[X] + [i for i in Pi] + ['count']]
    xy = joint_counts.groupby([X] + [i for i in Pi]).sum()
    if not Pi:  # if Pi is an empty set
        family_score = -1 * centropyd(np.array(xy['count']), np.array([]))
    else:  # get counts of joint distribution of Pi
        Pi_counts = counts[[i for i in Pi] + ['count']]
        # (Pi_counts)
        y = Pi_counts.groupby([i for i in Pi]).sum()
        # print(y)
        # compute the plug-in estimator of H(X | Pi)
        family_score = -1 * centropyd(np.array(xy['count']), np.array(y['count']))

    return family_score

def entropyd_memory_saving(count, base=2):
    """ Discrete entropy estimator
        sx is a list of samples
    """
    proba = count.astype(float) / sum(count)
    # Avoid 0 division; remove probabilities == 0.0 (removing them does not change the entropy estimate as 0 * log(1/0) = 0.
    proba = proba[proba > 0.0]
    return np.sum(proba * np.log(1. / proba)) / math.log(base)


def entropyd(count, base=2):
    """ Discrete entropy estimator
        count is a list of samples
    """
    proba = count.astype(float) / sum(count)
    # Avoid 0 division; remove probabilities == 0.0 (removing them does not change the entropy estimate as 0 * log(1/0) = 0.
    proba = proba[proba > 0.0]
    return np.sum(proba * np.log(1. / proba)) / math.log(base)

def centropyd(xy, y, base=2):
    """ The classic K-L k-nearest neighbor continuous entropy estimator for the
        entropy of X conditioned on Y.
    """

    if np.size(y) == 0:
        return entropyd(xy, base)

    return entropyd(xy, base) - entropyd(y, base)

def run_gobnilp(prob_name, k, set=False):

    if set:
        settings = open('./eBNSL/networks/settings/' + prob_name + '.opt', 'w+')  # create settings file
        # settings file includes: maximal number of parents k, no variable names in the score file, path to write solution
        settings.write('gobnilp/scoring/palim = ' + str(
            k) + '\ngobnilp/scoring/arities = FALSE\ngobnilp/outputfile/solution = "./networks/' + prob_name + '.opt"')
        settings.close()

    os.system('./eBNSL/gobnilp/bin/gobnilp -f=jkl -g=./networks/settings/' + prob_name + '.opt  ./eBNSL/scores/' + prob_name)

def run_ebnsl(prob_name, theta, k, id='""'):
    os.system('./eBNSL/run_score.sh ' + prob_name + ' ' + str(theta) + ' ' + str(k) + ' ' + id)

def update_constraint_file(path, f, vars, k):
    c = open(path, 'a+')

    v = f.get_child()
    P = f.get_parents()

    for p in P:
        c.write(str(v) + '<-' + p + '\n')  # add all v's parents to constraint file
    if len(P) < k:  # if parent set has less than k variables
        P_not = set(vars) - (set(P).union({v}))
        for pn in P_not:
            c.write('~' + v + '<-' + pn + '\n')
    c.close()

def families_intersection(vars, d, file_path):
    f = open(file_path,'r')
    lines = f.read().splitlines()



    if lines:
        m = int(lines[-1]) + 1  # number of networks in the file
    else:
        m = 1
    # read optimal network
    o = open(file_path + '.opt', 'r')
    opt_net = o.read().splitlines()
    lines = opt_net + ['\n'] + lines  # merge optimal network with other solutions


    EC_as_families = dict()  # equivalence classes represented as a set of families
    EC_as_lists = dict()  # equivalence classes represented as lists of edges and v-structure
    net = list()  # network as a list of families

    i = 0
    num_nets = 0


    # read networks from inputted file
    while i < (len(lines) - 1):
        g = Graph()
        if 'BN' not in lines[i]:  # still reading the network
            var = lines[i].partition("<")[0]
            parents = [p for p in (lines[i].split('-')[1]).split(',')[:-1]]
            g.add_family(Family(var, parents))
            i = i + 1

        else:
            num_nets = num_nets + 1
            curr_eq = g.getEC()
            # check if equivalence class already exists
            eq_exists = False
            for key in EC_as_lists:
                if curr_eq.compare(EC_as_lists[key]):
                    EC_as_families[key].union(g.get_graph())
                    eq_exists = True
                    break

            if not eq_exists:
                new_key = len(EC_as_lists)
                EC_as_lists[new_key] = curr_eq
                EC_as_families[new_key] = set()
                EC_as_families[new_key].union(g.get_graph())

            i = i + 3

    # return a set of families which is the intersection of all ECs
    num_ECs = len(EC_as_families)
    if len(EC_as_families) > 1:
        intersect = EC_as_families[0].intersection(EC_as_families[1])
        for i in range(2, len(EC_as_families)):
            intersect = intersect.intersection(EC_as_families[i])
    else:
        intersect = EC_as_families[0]
    print(EC_as_lists)
    return intersect, m, num_ECs
