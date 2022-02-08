#!/bin/bash

if [ "$#" -ne 4 ]
then
echo "Usage: ./run_score.sh <scorename> <bf>
where <scorename> is the name of the scoring file in ./scores/,
and <bf> is the desired Bayes factor,
e.g., ./run_score.sh wine.BIC.3 for ./scores/wine.BIC.3 and a Bayes factor of 3.
"
exit 1
fi

mkdir -p ./eBNSL/networks/settings

echo "#GOBNILP parameters for finding the optimal network
gobnilp/outputfile/solution = \"./eBNSL/networks/$1$4.opt\"
gobnilp/dagconstraintsfile = \"./eBNSL/constraints/constraints$1$4\"
gobnilp/scoring/palim = $3
gobnilp/outputfile/solutionavg = \"\" " > ./eBNSL/networks/settings/$1$4.opt

./eBNSL/gobnilp/bin/gobnilp -f=jkl -g=./eBNSL/networks/settings/$1$4.opt ./eBNSL/scores/$1$4

score=$(tail -1 ./eBNSL/networks/$1$4.opt | awk '{print $4}')
limit=$(echo "$score-$2" | bc -l)

echo "#GOBNILP parameters for collecting networks
gobnilp/countsols = TRUE
gobnilp/countsols/collect = TRUE
gobnilp/countsols/sollimit = 150000
gobnilp/objlimit = $limit
gobnilp/outputfile/countsols = \"./eBNSL/networks/$1$4\"
gobnilp/dagconstraintsfile = \"./eBNSL/constraints/constraints$1$4\"
gobnilp/scoring/palim = $3" > ./eBNSL/networks/settings/$1$4

./eBNSL/gobnilp/bin/gobnilp -g=./eBNSL/networks/settings/$1$4 -f=jkl ./eBNSL/scores/$1$4
	