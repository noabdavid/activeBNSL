import pandas as pd
import sample_size
import math
import random
import utils
from graph import Graph


def activeBNSL(d, k, epsilon, delta, prob_name, epsilon1=0.5 ** 5, Ma=None, Mb=None):
    id = utils.generate_id()  # this id is used for all the files of a specific run

    # create constraints file
    c = open('./eBNSL/constraints' + id, 'w+')
    c.close()
    # create log file
    log = open('log' + id, 'w+')
    log.write('epsilon = ' + str(epsilon) + '\nr = ' + str(d / epsilon) + '\n')
    score_path = './eBNSL/scores/' + prob_name + id
    network_path = './eBNSL/networks/' + prob_name + id

    # initialization
    model, node_names, states = utils.load_network(prob_name)
    active_vars = node_names.copy()
    Accept = Graph()
    N_curr = 0
    t = 1

    epsilon_t = epsilon1
    T = math.ceil(math.log2((2 * d * epsilon_t) / epsilon))
    log.write('Number of rounds T = ' + str(T) + '\n')

    counts = pd.DataFrame(columns=node_names + ['count'])

    if Ma is None or Mb is None:
        print('Computing support..')
        Ma, Mb = utils.support(states, k)


    while epsilon_t > (epsilon / len(active_vars)):

        log.write('Accuracy level in round ' + str(t) + ': ' + str(epsilon_t) + '\n')
        print('Computing sample size..')
        delta_t = delta/(T * len(active_vars) * ((d - 1) ** k))
        N_next = utils.N(epsilon_t/2, delta_t, Ma, Mb)

        print('Sampling data..')
        counts = counts.append(utils.sample_subsets(k, model, states, N_next - N_curr, active_vars))
        log.write('Number of samples taken per family in round ' + str(t) + ' : ' + str(N_next - N_curr) + '\n')
        N_curr = N_next
        print('Computing scores...')
        utils.update_score_file(d, active_vars, k, counts, min(1, t - 1), score_path)

        family_accepted = True

        while family_accepted:
            theta = len(active_vars) * epsilon_t
            log.write('Gap = ' + str(theta) + '\n')

            # run eBNSL with scores file saved in score_path to collect networks within a factor of theta
            utils.run_ebnsl(prob_name, theta, k, id)

            intersect, num_sols, num_ECs = utils.families_intersection(node_names, d, network_path)

            log.write('Families in intersection: ' + str(intersect) + '\n')
            log.write('Number of solutions found: ' + str(num_sols) + '\nNumber of equivalence calsses: ' + str( num_ECs) + '\n')
            candidates = intersect - Accept.get_graph()
            log.write('Candidate families for acceptance: ' + str(candidates) + '\n')
            family_accepted = bool(candidates)
            if family_accepted:
                f = random.choice(tuple(candidates))  # accepted family is chosen randomly
                utils.update_constraint_file('constraints' + id, f, node_names.copy(), k)

                candidates.remove(f)
                Accept.add_family(f)
                log.write('Family ' + str(f) + ' was accepted...\n')

                v = f.get_child()
                active_vars.remove(v)
                log.write('Variable ' + str(v) + ' is no longer active\n')

        if Accept.num_families() == d:
            log.write('BN: ' + str(sorted(list(Accept), key=lambda x: x[0])) + '\nTotal number of samples: ' + str(sum(list(counts['count']))))
            log.close()
            return Accept, sum(list(counts['count'])), d

        t = t + 1
        epsilon_t = epsilon_t / 2

    epsilon_last = epsilon/len(active_vars)
    log.write('epsilon_last = ' + str(epsilon_last) + ', delta_last = ' + str(delta / (T * len(active_vars) * ((d - 1) ** k))) + '\n')
    N_next = sample_size.sample_size(epsilon_last / 2, delta / (T * len(active_vars) * ((d - 1) ** k)), Ma, Mb)
    log.write('Number of samples taken per family in the last round: ' + str(max(0,N_next - N_curr)) + '\n')

    if N_next - N_curr > 0:  #  added to fix a bug in the algorithm
        counts = counts.append(utils.sample_subsets(k, model, states, N_next - N_curr, active_vars))
    # run gobnilp with scores file saved in file_path
    utils.update_score_file(d, active_vars, k, counts, min(1, t-1), score_path)
    utils.run_gobnilp(prob_name + id, k)

    G_hat = Graph('./eBNSL/networks/' + prob_name + id + '.opt', node_names)
    log.write('BN: ' + str(sorted(list(G_hat), key=lambda x: x[0])))
    log.close()
    #return output, len(data.index)
    return G_hat, sum(list(counts['count'])), d - len(active_vars)


#print('0.9-optimal network: ' + str(activeBNSL(d=5, k=2, T=3, epsilon=0.9, delta=0.05, errors=[2.6, 2], prob_name='cancer')))
#G_acive, N, num_accepted = activeBNSL(d=8, k=2, epsilon=0.00390625, delta=0.05, prob_name='vGammaStable8')
#print('G_active = ' + str(G_acive) + '\nN_active = ' + str(N) + '\nnum_accepted = ' + str(num_accepted))

