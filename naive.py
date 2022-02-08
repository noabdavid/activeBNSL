import sample_size
import families
import utils


def naive(d, k, epsilon, delta, prob_name, penalty=False, Ma=None, Mb=None):
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
    # initialize
    # read network from file
    model, node_names, states = utils.loadNetwork(prob_name)
    id = utils.generateId()  # this id is used for all the files of a specific run

    epsilon_tag = epsilon / (2*d)
    delta_tag = delta / utils.computeNumberOfFamilies(d, k)
    if Ma is None or Mb is None:
        Ma, Mb = sample_size.support(states, k)
    N = sample_size.sample_size(epsilon_tag, delta_tag, Ma, Mb)
    print('Number of samples from each subset of size  ' + str(k+1) + ': ' + str(N))

    # sample
    print('Sampling..')
    counts = families.sample_subsets(k, model, states, N)

    # compute scores
    print('Computing local scores..')
    scores_path = './eBNSL/scores/' + prob_name + id
    families.compute_scores(d, node_names, k, counts, 0, scores_path, penalty)

    # run gobnilp to discover optimal BN based on computed scores
    utils.runGOBNILP(prob_name + id, k, True)

    # return best network
    G_hat = families.bn_file_to_family_set('./eBNSL/networks/' + prob_name + id + '.opt', node_names)
    return G_hat, sum(list(counts['count']))

