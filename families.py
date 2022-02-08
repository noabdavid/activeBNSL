import itertools
import pandas as pd
import numpy as np
import math
import scipy.special
import scipy.stats
from pgmpy.sampling import BayesianModelSampling
from pgmpy.inference import VariableElimination
from itertools import chain, combinations




def sample_subsets(k, model, states, N, active_vars=[]):

    """ returns the counts as a dataframe of N samples per subset of size k+1 
...
 Parameters
        ----------
        k : int
            maximal in-degree
        model      : pgmpy Bayesian Model
                    Bayesian network to sample from
        states     : dict
                    states[i] is the list of possible states of node i
        N          : int
                    number of smaples to take from each subset of size k+1
        active_vars: list, optional
                    list with the name of active nodes
"""
    infer = VariableElimination(model)  # query object
    node_names = list(model.nodes())
    sample = pd.DataFrame(columns=node_names + ['count'])

    for S in itertools.combinations(node_names, k + 1):
        # given a set of active nodes, sample only subsets in which at least one variable is active
        if not active_vars or bool(set(S).intersection(set(active_vars))):
            q = infer.query(variables=list(S), evidence={})  # joint distribution of variables
            #print('joint probability of ' + str(S) + ': ' + str((q.values).flatten()))
            curr_states = [states[s] for s in q.variables]
            rng = np.random.default_rng()
            counts = rng.multinomial(N, (q.values).flatten())  # sample of size N from q
            #print(q)
            curr_sample = pd.DataFrame(list(itertools.product(*curr_states)), columns=q.variables)
            curr_sample['count'] = counts
            #print(curr_sample)
            sample = pd.concat([curr_sample, sample]).fillna(np.nan)
            #print(sample)
    return sample
        
# ======entropy functions from npeet=========================================

def entropyd(count, base=2):
    """ Discrete entropy estimator
        sx is a list of samples
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

# ===========================================================================

def compute_scores(d, vars, k, counts, exists_file, file_path, penalty):
""" write the score file. The file contains scores for each possible family
...
 Parameters
        ----------
        d           : int
                      number of nodes
        vars        : list
                      list with names of nodes
        k           : int
                      maximal in-degree
        counts      : DataFrame
                      counts of samples (output of samples_subsets(...))
        exists_file : bool
                      True if there already exists a previous score file for this run
        file_path   : str
                      location of file to save/update
        penalty     : bool
                      True to include the penalty term in the score computation
"""
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
                    # get counts of joint distribution of X, Pi
                    joint_counts = counts[[X] + [i for i in Pi] + ['count']]
                    xy = joint_counts.groupby([X] + [i for i in Pi]).sum()
                    #print(xy)

                    if not Pi:  # if Pi is an empty set
                        family_score = -1 * centropyd(np.array(xy['count']), np.array([]))
                        baseline = family_score
                    else:  # get counts of joint distribution of Pi
                        y = xy.groupby([i for i in Pi]).sum()
                        # compute the plug-in estimator of H(X | Pi)
                        family_score = -1 * centropyd(np.array(xy['count']), np.array(y['count']))
                        if penalty:
                            family_score = family_score - len(Pi) * 0.001 * abs(baseline)
                    # go to the line where the score should be updated
                    if exists_file:

                        if not Pi:
                            st = [str(k_tag)] + ['']
                        else:
                            st = [str(k_tag)] + [x for x in Pi]
                        while st != content[i].split(' ')[1:]:
                            #print('searching in ' + content[i])
                            i = i + 1
                        # rewrite line with updated score
                        #print('old score = ' + content[i].split(" ")[0])
                        #print('new score = ' + str(family_score))
                        content[i] = str(family_score) + ' ' + str(k_tag) + ' ' + ' '.join(Pi)
                        #print('new line: ' + content[i])
                        #print('rewriting ' + content[i])
                    else:
                        num_parent_sets += 1
                        content.insert(i + num_parent_sets, str(family_score) + ' ' + str(k_tag) + ' ' + ' '.join(Pi))

        if not exists_file:
            content.insert(i, X + ' ' + str(num_parent_sets))
            i = i + num_parent_sets + 1

    if exists_file:
        f = open(file_path, 'w')

    f.write('\n'.join(content))


# ===========================================================================
# find families that are in the intersection of equivalence classes of networks in the inputed file

def families_intersection(vars, d, file_path, k, active_vars):
""" find families that are in the intersection of equivalence classes
...
 Parameters
        ----------
        vars        : list
                      list with names of nodes
        d           : int
                      number of nodes
        file_path   : str
                      location of output file of eBNSL
        k           : int
                      maximal in-degree
        active_vars: list, optional
                    list with the name of active nodes
"""
    f = open(file_path,'r')
    lines = f.read().splitlines()



    if lines:
        m = int(lines[-1]) + 1  # number of networks in the file
    else:
        m = 1

    # read optimal network
    o = open(file_path + '.opt', 'r')
    opt_net = o.read().splitlines()
    lines = opt_net + ['\n'] + lines

    # merge optimal network with other solutions
    ECs = dict()
    net = set()  # network as a list of families
    struct = np.zeros((d, d))  # network as adjacency matrix
    intersect = np.zeros((d, d))  # the intersection of the ECs represented as adjacency matrix
    i = 0
    num_nets = 0
    vars_enum = {}
    for idx, val in enumerate(vars):
        vars_enum[val] = idx
    active_vars_enum = [vars_enum[val] for val in active_vars]

    # read networks from inputted file
    while i < (len(lines) - 1):

        if 'BN' not in lines[i]:  # still reading the network
            var = lines[i].partition("<")[0]
            parents = [p for p in (lines[i].split('-')[1]).split(',')[:-1]]
            net.add((var, tuple(parents)))  # add family
            if parents:  # add edge to matrix
                inds = [value for key, value in vars_enum.items() if key in parents]
                # enumeration for the variables names
                struct[vars_enum[var]][inds] = 1
            i = i + 1

        else:
            num_nets = num_nets + 1
            # compute network's equivalence class. If already observed, add families of the network to the EC representation, else, create new EC
            curr_ec = compute_equivalence_class(d, struct.transpose())
            if curr_ec not in ECs.keys():
                ECs[curr_ec] = computeEssentialGraph(d, struct.transpose().copy(), active_vars_enum)
                if len(ECs) == 1:
                    intersect = ECs[curr_ec]
                else:
                    intersect = ecsIntersection(intersect, ECs[curr_ec], active_vars_enum)
                    print('intersection = \n' + str(intersect))
                    if isEmpty(intersect):
                        break

            struct = np.zeros((d, d))
            net = set()
            i = i + 3

    # return a set of families which is the intersection of all ECs
    num_ECs = len(ECs)
    intersect = ecsIntersectionToFamilies(intersect, vars, k)
    return intersect, m, num_ECs

def isEmpty(intersect):
    unique, counts = np.unique(intersect[0, :], return_counts=True)
    d = dict(zip(unique, counts))
    return -1 in d.keys() and d[-1] == len(intersect)

def computeEssentialGraph(d, g, active_vars):
    g_prev = np.copy(g)
    while True:
        map = protectedEdgeMap(d, g)

        for i,j in np.transpose(g.nonzero()):
            if j in active_vars and i in active_vars and not map[i,j]:
               g[j,i] = 1  # turn i->j to i-j if it is not protected and if j is an active variable
        if np.array_equal(g_prev, g):
            return g
        g_prev = np.copy(g)

def protectedEdgeMap(d, g):
    map = np.zeros((d,d))
    for a,b in np.transpose(g.nonzero()):
        if g[b, a] == 0.0:
            print('examining edge ' + str(a) + '->' + str(b))
            isProtected = False
            # case 1: c->a->b
            for c in np.where(g[:, a] == 1.0)[0]:
                isProtected = not g[a, c] and not g[c,b] and not g[b,c]
                if isProtected:
                    print(str(a) + '->' + str(b) + ' is protected from case 1, following edge ' + str(c) + '->' + str(a))
                    break
            # case 2+3: a->b<-c or c<-a->b and c->b
            for c in np.where(g[:, b] == 1.0)[0]:
                if isProtected:
                    break
                isProtected = (not c == a) and ((not g[c,a] and not g[a,c]) or (g[a,c] and not g[c, a]))
                if (isProtected):
                    print(str(a) + '->' + str(b) + ' is protected from case 2 or 3')
                    print('explanation:')
                    if (not g[c,a] and not g[a,c]):
                        print('there is not edge between ' + str(c) + ' and ' + str(a))
                    elif g[a,c]:
                        print('there is an edge from ' + str(a) + ' to ' + str(c))
            # case 4:
            for (c1, c2) in itertools.combinations(np.where(g[:, b] == 1.0)[0], 2):
                if isProtected:
                    break
                isProtected = (not g[b, c1] and not g[b, c2]) and g[c1, a] and g[a, c1] and g[c2, a] and g[a, c2]
                if (isProtected):
                    print(str(a) + '->' + str(b) + ' is protected from case 4')
            if not isProtected:
                print(str(a) + '->' + str(b) + ' is not protected')
            map[a][b] = isProtected
    return map




def ecsIntersection(e1, e2, active_vars):
    d = len(e1)
    e = np.zeros((d, d))
    for i in range(d):
        if i not in active_vars:
            e[:, i] = e1[:, i]

    # check for compelled edges
    for i, j in np.transpose(e1.nonzero()):
        if e1[j, i] == 0 and e2[i, j] == 1.0:
            e2[j, i] = 0

    for i, j in np.transpose(e2.nonzero()):
        if e2[j, i] == 0 and e1[i, j] == 1.0:
            e1[j, i] = 0

    for i in active_vars:
        if np.array_equal(e1[:, i], e2[:, i]):
            e[:, i] = e1[:, i].copy()
        else:
            e[0, i] = -1
    return e

def ecsIntersectionToFamilies(e, vars, k):
    intersection = set()
    d = len(vars)
    for i in range(d):
        col = e[:,i]
        if not -1 in col:
            connected = np.where(col == 1.0)[0]
            compelled_parents = []
            for c in connected:
                if e[i,c] == 0:
                    compelled_parents.append(c)
            if len(compelled_parents) == k:
                par = [vars[l] for l in compelled_parents]
                par.sort()
                intersection.add((vars[i], tuple(par)))
            else:
                k_tag = k - len(compelled_parents)
                for p in powerset([item for item in connected if item not in compelled_parents]):
                    if len(p) > k_tag:
                        break
                    print(type(compelled_parents))
                    
                    if p:
                        p = [vars[o] for o in p]
                        par = [vars[l] for l in compelled_parents]
                        if par:
                            par = par + p
                        else:
                            par = p
                        #print(par)
                        par.sort()
                        intersection.add((vars[i], tuple(par)))
                    else:
                        par = [vars[l] for l in compelled_parents]
                        if par:
                            par.sort()
                            intersection.add((vars[i], tuple(par)))
    return intersection



def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))



# compute the equivalence class of the given structure represented as a set of edges and a set of v-structure
# d     : integer, total number of variables
# struct: numpy 2x2 array such that struct[i][j] = 1 if and only if i is a parent of j
def compute_equivalence_class(d, struct):
    #edges = set()
    #v_struct = set()
    edges = list()
    v_struct = list()

    for i in range(d):
        parents = np.where(struct[:,i])[0] # parents of variable i
        if len(parents) == 1:
            #edges.update({tuple(sorted((i, parents[0])))})
            edges.append(tuple(sorted((i, parents[0]))))
        else:
            for pair in itertools.combinations(parents, 2):
                p1 = pair[0]
                p2 = pair[1]
                if not struct[p1][p2] and not struct[p2][p1]: # p1->i<-p2 is a v-structure
                    #v_struct.update({(i, pair)})
                    v_struct.append((i, pair))
                edges.append(tuple(sorted((i, p1))))
                edges.append(tuple(sorted((i, p2))))

    return (tuple(edges), tuple(v_struct))

def bn_file_to_family_set(file_path, vars):
    f = open(file_path, 'r')
    lines = f.read().splitlines()
    dag = set()
    for i in range(len(vars)):
        parents = [p for p in (lines[i].split('-')[1]).split(',')[:-1]]
        dag.update({(vars[i], tuple(parents))})  # add family
    return dag






