import os
from pgmpy.readwrite import BIFReader
import uuid
import math

def loadNetwork(net_name):
    reader = BIFReader('./BIF/' + net_name + '.bif')
    states = reader.get_states()  # get all node states
    node_names = list(states.keys())  # get node names
    model = reader.get_model()
    return model, node_names, states

def generateId():
    return str(uuid.uuid1())

def binom(n, k):
    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)

def runEBNSL(prob_name, theta, k, id='""'):
    os.system('./run_score.sh ' + prob_name + ' ' + '{:.10f}'.format(theta) + ' ' + str(k) + ' ' + id)

def runGOBNILP(prob_name, k, set=False):
    if set:
        settings = open('./eBNSL/networks/settings/' + prob_name + '.opt', 'w+')  # create settings file
        # settings file includes: maximal number of parents k, no variable names in the score file, path to write solution
        settings.write('gobnilp/scoring/palim = ' + str(
            k) + '\ngobnilp/scoring/arities = FALSE\ngobnilp/outputfile/solution = "./eBNSL/networks/' + prob_name + '.opt"')
        settings.close()

    os.system('./eBNSL/gobnilp/bin/gobnilp -f=jkl -g=./eBNSL/networks/settings/' + prob_name + '.opt  ./eBNSL/scores/' + prob_name)

def computeNumberOfFamilies(d, k, c=None):
    sum_coeff = 0
    for i in range(0, k+1):
        sum_coeff = sum_coeff + binom(d-1, i)
    if not c:
        return d * sum_coeff
    return c * sum_coeff
