#!/bin/bash

# Define min_n and max_n pairs
declare -a min_max_pairs=("6 10" "11 15" "16 20" "21 25")

# Define options (including an empty option for Random suffix)
declare -a options=("" "--var" "--skewed" "--marginal")

# Loop through each min_n and max_n pair
for pair in "${min_max_pairs[@]}"; do
    read min_n max_n <<< ${pair}
    
    # Loop through each option
    for option in "${options[@]}"; do
        suffix="random" # Default suffix
        if [ ! -z "$option" ]; then
            # Extract option name without dashes and capitalize it
            suffix=$(echo $option | cut -d' ' -f1 | sed 's/--//g' | awk '{print $0}')
        fi
        
        # Generate the command for the training dataset
        python generate_formula.py 50000 datasets/SAT_${min_n}_${max_n}_${suffix}.txt --min_n ${min_n} --max_n ${max_n} ${option}
        
        # Generate the command for the test dataset
        python generate_formula.py 2000 datasets/SAT_${min_n}_${max_n}_${suffix}_Test.txt --min_n ${min_n} --max_n ${max_n} --seed 0 ${option}
    done
done
