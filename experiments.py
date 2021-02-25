# general functions for experiments
import naive
import activeBNSL
import utils
from graph import Graph


# compare the naive and the active algorithms
def alg_compare(d, k, epsilon, delta, prob_name):
    # run naive algorithm
    G_naive, N_naive = naive.naive(d, k, epsilon, delta, prob_name)

    # run active algorithm
    G_active, N_active, num_accepted = activeBNSL.activeBNSL(d, k, epsilon, delta, prob_name)

    return G_naive, N_naive, G_active, N_active, num_accepted

def expeirment(iter, d, r, delta, prob_name):
    # load BayeisnaModel from file and get family representation
    model, states, _ = utils.load_network(prob_name)
    true_graph = Graph(model, d)

    # get results from naive and active algorithms
    epsilon = float(d) / r
    k = 2
    G_naive, N_naive, G_active, N_active, num_accepted = alg_compare(d=d, k=k, epsilon=epsilon, delta=delta, prob_name=prob_name)

    # compute true epsilons
    true_score = true_graph.compute_score(model)
    naive_true_epsilon = true_score - G_naive.compute_score(model)
    active_true_epsilon = true_score - G_active.compute_score(model)
    # write results to file
    f = open('./results/' + prob_name + str(iter) + '_r' + str(r) + '.results', 'w+')
    f.write(str(N_naive) + '\n' + str(N_active) + '\n' + str(epsilon) + '\n' + str(naive_true_epsilon) + '\n' + str(active_true_epsilon) + '\n' + str(G_naive) + '\n' + str(G_active) + '\n' + str(num_accepted))


