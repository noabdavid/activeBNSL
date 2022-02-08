import pandas as pd
import sample_size
import families
import utils
import math
import random


def activeBNSL(d, k, epsilon, delta, prob_name, penalty=False, Ma=None, Mb=None):
    """ 
 Parameters
        ----------
        d           : int
                      number of nodes
        k           : int
                      maximal in-degree
        epsilon     : float
                      accuracy level
        delta       : float
                      confidence level
        prob_name   : str
                      name of Bayesian network as saved in BIF folder
        penalty     : bool, optional
                      True to include the penalty term in the score computation
        Ma          : int, optional
                      maximal support among nodes. If not given, this would be computed.
        Mb          : int, optional
                      maximal support among subsets of size k. If not given, this would be computed.
"""
    id = utils.generateId()  # this id is used for all the files of a specific run
    score_path = './eBNSL/scores/' + prob_name + id
    network_path = './eBNSL/networks/' + prob_name + id
    constraints_path = './eBNSL/constraints/constraints' + prob_name + id

    # create constraints file
    print('creating constraint file...')
    c = open(constraints_path, 'w+')
    c.close()
    # create log file
    log = open('log' + id, 'w+')
    log.write('epsilon = ' + str(epsilon) + '\nr = ' + str(d / epsilon) + '\n')


    model, node_names, states = utils.loadNetwork(prob_name)
    active_vars = node_names.copy()
    Accept = set()
    N_curr = 0
    t = 1
    epsilon_t = epsilon
    T = math.ceil(math.log2(2 * d))
    log.write('Number of rounds T = ' + str(T) + '\n')
    counts = pd.DataFrame(columns=node_names + ['count'])
    num_families = utils.computeNumberOfFamilies(d, k)

    if Ma == None or Mb == None:
        print('Computing support..')
        Ma, Mb = sample_size.support(states, k)
        

    delta_tag = delta / (T * num_families)

    while epsilon_t > (epsilon / len(active_vars)):

        log.write('Accuracy level in round ' + str(t) + ': ' + str(epsilon_t) + '\n')
        print('Computing sample size..')
        N_next = sample_size.sample_size(epsilon_t/2, delta_tag, Ma, Mb)
        

        print('Sampling data..')
        counts = pd.concat([counts, families.sample_subsets(k, model, states, N_next - N_curr, active_vars)])
        log.write('Number of samples taken per family in round ' + str(t) + ' : ' + str(N_next - N_curr) + '\n')
        N_curr = N_next

        print('Computing scores...')
        families.compute_scores(d, active_vars, k, counts, min(1, t - 1), score_path, penalty)

        family_accepted = True

        while family_accepted:
            theta = len(active_vars) * epsilon_t

            # run eBNSL with scores file saved in file_path to collect networks within a factor of theta
            utils.runEBNSL(prob_name, theta, k, id)
            print('Gap = ' + str(theta))
            print(('./run_score.sh ' + prob_name + ' ' + '{:.10f}'.format(theta) + ' ' + str(k) + ' ' + id))
            log.write('Gap = ' + str(theta) + '\n')

            intersect, num_sols, num_ECs = families.families_intersection(node_names, d, network_path, k, active_vars)
            log.write('Families in intersection: ' + str(intersect) + '\n')
            log.write('Number of solutions found: ' + str(num_sols) + '\nNumber of equivalence calsses: ' + str(
                num_ECs) + '\n')
            if num_sols == 1:
                Accept = Accept.union(intersect)
                log.write('Accept: ' + str(Accept) + '\n')
                family_accepted = False

            else:
                candidates = intersect - Accept
                log.write('Candidate families for acceptance: ' + str(candidates) + '\n')
                family_accepted = bool(candidates)
                if family_accepted:
                    f = random.choice(tuple(candidates))  # accepted family is chosen randomly
                    updateConstraintFile(constraints_path, f, node_names, k)
                    candidates.remove(f)
                    Accept.add(f)
                    log.write('Family ' + str(f) + ' was accepted...\n')
                    print(active_vars)
                    print(f[0])
                    active_vars.remove(f[0])
                    log.write('Variable ' + str(f[0]) + ' is no longer active\n')

        if len(Accept) == d:
            log.write('BN: ' + str(sorted(list(Accept), key=lambda x: x[0])) + '\nTotal number of samples: ' + str(sum(list(counts['count']))))
            log.close()
            return Accept, sum(list(counts['count'])), d

        t = t + 1
        epsilon_t = epsilon_t / 2

    epsilon_T = epsilon/len(active_vars)
    log.write('epsilon_T = ' + str(epsilon_T) + ', delta_T = ' + str(delta_tag) + '\n')
    N_next = sample_size.sample_size(epsilon_T / 2, delta_tag, Ma, Mb)
    log.write('Number of samples taken per family in the last round: ' + str(max(0,N_next - N_curr)) + '\n')
    if N_next - N_curr > 0:  #  added to fix a bug in the algorithm
        counts = pd.concat([counts, families.sample_subsets(k, model, states, N_next - N_curr, active_vars)])

    # run gobnilp with scores file saved in file_path
    families.compute_scores(d, active_vars, k, counts, min(1, t - 1), score_path, penalty)
    utils.runGOBNILP(prob_name + id, k)
    G_hat = families.bn_file_to_family_set('./eBNSL/networks/' + prob_name + id + '.opt', node_names)
    output = G_hat.union(Accept)
    log.write('BN: ' + str(sorted(list(output), key=lambda x: x[0])))
    log.close()
    return output, sum(list(counts['count'])), len(Accept)

def updateConstraintFile(path, f, vars, k):
    c = open(path, 'a+')

    v = f[0]
    P = f[1]

    for p in P:
        c.write(str(v) + '<-' + p + '\n')  # add all v's parents to constraint file
    if len(P) < k:  # if parent set has less than k variables
        P_not = set(vars) - (set(P).union({v}))
        for pn in P_not:
            c.write('~' + v + '<-' + pn + '\n')
    c.close()


