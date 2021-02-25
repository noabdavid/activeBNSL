import utils
from graph import Graph


def naive(d, k, epsilon, delta, prob_name, Ma=None, Mb=None):

    model, node_names, states = utils.load_network(prob_name)
    id = utils.generate_id()  # this id is used for all the files of a specific run

    # initialize
    epsilon_tag = epsilon / (2*d)
    delta_tag = delta / (d ** (k+1))
    if Ma is None or Mb is None:
        Ma, Mb = utils.support(states, k)
    N = utils.N(epsilon_tag, delta_tag, Ma, Mb)
    print('Number of samples from each subset of size  ' + str(k+1) + ': ' + str(N))

    # sample
    print('Sampling..')
    counts = utils.sample_subsets(k, model, states, N)

    # compute scores
    print('Computing local scores..')
    scores_path = './eBNSL/scores/' + prob_name + id
    utils.update_score_file(d, node_names, k, counts, 0, scores_path)

    # run gobnilp to discover optimal BN based on computed scores
    utils.run_gobnilp(prob_name + id, k, True)

    # return best network
    G_hat = Graph('./eBNSL/networks/' + prob_name + id + '.opt', node_names)
    return G_hat, sum(list(counts['count']))



#G_naive, N = naive(d=11, k=2, epsilon=0.0859375, delta=0.05, prob_name = 'vGammaStable11')
#print('G = ' + str(G_naive) + '\nNumber of samples taken: ' + str(N))
#print('G_naive: \n' + str(naive(d=5, k=2, epsilon=2, delta=0.05, prob_name = 'cancer10K')))

