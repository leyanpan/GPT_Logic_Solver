# HF_SAT
Accompanying code for the paper "Can Transformers Reason Logically? A Study in SAT-Solving".

## Preparation
Since most experiments in the paper involves running/training models, it is highly recommended to run the repo on a GPU device with CUDA isntalled.
Install dependencies with environment.yml file:
```bash
conda env create -f environment.yml
conda activate HF_SAT
```

## Playing with the compiled model
We provide a file for interacting with a compiled model that outputs a Chain-of-Thought for SAT solving. The model takes as input a 3-SAT formula encoded in DIMACS format and outputs a sequence that ends with SAT if the formula is satisfiable, or UNSAT otherwise. For the detailed Chain-of-Thought, you can refer to the paper.
```bash
cd theory
python generate.py
```
The script will prompt you to enter a SAT problem in DIMACS format. You can find some examples in the `theory/basic_test.ipynb`. Specifically, you may start with the following examples (Each line represent a separate prompt):
```
1 3 0 -1 2 -3 0 4 -1 0 1 -3 4 0 -2 -3 -4 0 3 -4 -1 0 1 -3 -4 0 3 -1 4 0
```
```
6 -8 10 0 2 -6 -7 0 8 -7 3 0 -3 1 -2 0 -2 -10 4 0 1 -6 -2 0 -8 4 -9 0 1 10 -5 0 8 4 5 0 -7 -6 -4 0 6 3 10 0 1 7 -10 0 1 6 -3 0 7 10 -9 0 -2 1 4 0 -7 -9 4 0 9 6 -3 0 4 -2 -9 0 3 8 -1 0 -1 5 -3 0 -2 -3 -7 0 4 -2 8 0 -6 -9 -10 0 4 5 10 0 -7 3 -2 0 -6 7 4 0 -8 -1 -10 0 -7 8 10 0 -9 -7 -1 0 -6 -2 5 0 -6 5 -3 0 -6 -4 9 0 9 -10 7 0 8 -3 -7 0 7 -10 -5 0 1 2 -4 0 4 -8 3 0 8 -3 6 0 -8 -2 1 0 6 -3 -10 0 -1 -6 -2 0 6 -4 2 0 10 3 6 0
```
```
6 -7 -5 0 -4 -8 9 0 -1 -8 2 0 -3 6 1 0 5 -9 -10 0 -7 -1 9 0 -4 -6 10 0 -4 -8 7 0 -7 -2 -8 0 3 6 1 0 8 1 3 0 6 3 7 0 3 9 7 0 3 -9 -5 0 -1 3 2 0 5 3 -6 0 -10 7 4 0 8 -9 -10 0 1 -4 5 0 -2 10 7 0 -10 5 -2 0 -8 10 -7 0 -4 -5 -1 0 -10 -7 6 0 10 7 -1 0 -3 -6 7 0 4 6 -9 0 -9 -10 7 0 -7 -10 8 0 5 4 -2 0 -1 -3 4 0 -3 -8 1 0 7 9 10 0 4 3 2 0 10 2 3 0 9 -4 8 0 9 5 2 0 1 -2 -8 0 8 9 -1 0 1 -10 4 0 5 2 -1 0
```
By default, the compiled model allows 10 variables and 50 clauses. However, you can change this by passing command line arguments:
```bash
 python generate.py -v 20 -c 88 -l 1200
```
This will allow 20 variables, 88 clauses, and a sequence length of 1200.
Now you can play with the following larger 3-SAT instance:
```
-12 -9 6 0 11 14 -15 0 -3 2 -13 0 -11 6 13 0 15 -3 -14 0 6 -4 -1 0 -15 -13 11 0 2 -14 -5 0 14 15 -1 0 15 11 -8 0 12 9 6 0 -3 -2 1 0 -11 -3 8 0 -7 5 -14 0 -8 4 -15 0 1 -10 15 0 13 3 -1 0 12 -9 1 0 -2 7 14 0 3 7 -1 0 -7 2 13 0 -1 -8 13 0 7 4 13 0 6 -9 -7 0 -14 -9 -8 0 14 -11 13 0 -3 -10 1 0 6 15 -11 0 -3 -1 -10 0 -10 -9 11 0 2 13 5 0 8 -4 -12 0 -10 4 11 0 2 -13 -6 0 9 5 -15 0 8 4 -15 0 6 -15 -12 0 15 -11 10 0 5 -12 1 0 -4 -1 12 0 -7 5 11 0 9 -8 -1 0 -8 -9 4 0 -3 8 10 0 2 -5 -7 0 -13 -10 12 0 -8 5 1 0 -12 -3 -6 0 -11 -14 10 0 12 14 3 0 3 9 15 0 6 12 -15 0 9 3 -11 0 -3 13 6 0 7 1 -15 0 3 2 1 0 -12 3 11 0 -7 -1 -13 0 -10 -5 -14 0 -9 -14 1 0 5 -14 -7 0 -2 13 -5 0 14 3 -6 0
```

## Repository structure
- `theory/` contains the code for the compiler and the code for constructing the theoretical construction
- `sat_solver/` includes a customly written SAT solver for generating training datasets. It allows producion of synthetic Chain-of-Thoughts for SAT solving.
- `datasets/` includes download scripts for different datasets used in the paper.
- `configs/` contains training configurations for the models. You may train models with the following command:
```bash
python gpt_train.py configs/train_sat_6_10_marginal_rope_state_large.py
```




