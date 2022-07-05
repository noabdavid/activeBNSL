This project provides the implementations for the paper "Active Structure Learning of Bayesian Networks in an Observational Setting", Noa Ben-David and Sivan Sabato, Journal of Machine Learning Research, to appear. Preprint available at: https://arxiv.org/abs/2103.13796.


1. unzip eBNSL package with the command: unzip eBNSL.zip
2. navigate to eBNSL folder, and follow the README file to compile the package. 
3. in eBNSL folder, create an empty folder called 'networks' with the following command: mkdir networks
4. in eBNSL/networks/ folder, create an empty folder called 'settings' with the following command: mkdir settings
5. in eBNSL folder, create an empty folder called 'constraints' with the following command: mkdir constraints
6. create a Conda virtual environemnt using the command: conda create --name myenv
7 activate your environemnt with: conda activate myenv
8 install required packaged: numpy, pandas, scikit-learn, pgmpy (conda install -c ankurankan pgmpy). pgmpy might require manual installation of additional packages.
8. install numpy: conda install numpy
9. install pgmpy: conda install -c ankurankan pgmpy (pgmpy has the following dependencies: networkx, scipy, numpy, pytorch)
10. instal sklearn: conda install -c intel scikit-learn


NETOWRKS:

Both the naive algorithm and activeBNSL sample from a Bayesian network in a BIF format. 
To input one of the algorithms with a network, it must be saved in activeBNSL/BIF/.
See example networks in activeBNSL/BIF/ folder.
For further reading on BIF format see https://www.cs.washington.edu/dm/vfml/appendixes/bif.htm


THE NAIVE ALGORITHM:

To run the naive algorithm, use the following command from the python shell:
naive.naive(d=11, k=2, epsilon=0.1, delta=0.05, prob_name='vGammaStable11')

It is optional to input the maximal support size of a variable and a parent set. 
For example, if all variables are binary:
naive.naive(d=11, k=2, epsilon=0.1, delta=0.05, prob_name='vGammaStable11', Ma=2, Mb=4)

The function has two outputs: 
1. The learned graph, which is represented by a set of families.
2. The total number of samples taken.

ActiveBNSL:

To run activeBNSL, use th following command from the python shell:
activeBNSL.activeBNSL(d=8, k=2, epsilon=0.005, delta=0.05, prob_name='vGammaStable8')

The function has trhee outputs: 
1. The learned graph, which is represented by a set of families.
2. The total number of samples taken.
3. The number of accepted families

After activeBNSL terminates, there will be a log file in the main folder with details on what happend during the run.
There are some additional files that are generated and saved in differenet folders during the runs of both algorithms.
