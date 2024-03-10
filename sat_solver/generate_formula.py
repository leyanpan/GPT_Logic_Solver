# Generate formulas that are 1) all SAT and 2) have a lot of learned conflict clauses
# python generate_formula2.py datasets/SAT_CC.txt 1000

import numpy as np
import random
import argparse
from pysat.solvers import Solver
from tqdm import tqdm

def generate_k_iclause(n, k):
    vs = np.random.choice(n, size=min(n, k), replace=False).tolist()  # Convert to list to work with native Python integers
    return [int(v + 1) if random.random() < 0.5 else int(-(v + 1)) for v in vs]  # Ensure literals are Python integers

def generate_clause_assign(n, k, assignment, fst, snd, prob=0.75):
    # Skewed distribution: each literal have higher probability to have polarity same as a predefined "asignment"
    # and one half of variables is more likely to appear than another half.
    vs = []
    while len(vs) < min(n, k):
        if random.random() < prob:
            lit = np.random.choice(fst)
        else:
            lit = np.random.choice(snd)
        if lit not in vs:
            vs.append(lit)
    # Assign Polarity based on inverse of assignment
    inv = [v if assignment[v - 1] else -v for v in vs]
    inv = [int(i) if random.random() < 0.25 + prob / 2 else int(-i) for i in inv]
    return inv

def generate_pair(opts):
    n = random.randint(opts.min_n, opts.max_n)
    unsat = None
    sat = None
    while unsat is None or sat is None:
        if not opts.marginal:
            num_clauses = random.randint(int(n * opts.ratio_min), int(n * opts.ratio_max))
            i = 0
        if opts.skewed:
            assignment = [random.random() < 0.5 for _ in range(n)]
            perm = np.random.permutation(n) + 1
            fst = perm[:n // 2]
            snd = perm[n // 2:]
        clauses = []
        solver = Solver(name='g3')
        while True:
            k_base = 2 if random.random() < opts.p_k_2 else 1
            k = k_base + np.random.geometric(opts.p_geo)
            if not opts.skewed:
                iclause = generate_k_iclause(n, k)
            else:
                iclause = generate_clause_assign(n, k, assignment, fst, snd, opts.skew)
            clauses.append(iclause)
            solver.add_clause(iclause)
            if opts.marginal:
                if not solver.solve():
                    if len(clauses) <= n * opts.ratio_max and len(clauses) >= n * opts.ratio_min:
                        unsat = clauses
                        sat = clauses[:-1] + [iclause[:-1] + [-iclause[-1]]]
                    break
            else:
                i += 1
                if i >= num_clauses:
                    if solver.solve():
                        sat = clauses
                    else:
                        unsat = clauses
                    break
        solver.delete()
    return n, sat, unsat

def ch_form(iclauses):
    ret = ""
    for iclause in iclauses:
        for var in iclause:
            ret += str(var) + " "
        ret += "0 "
    return ret

def permute_formula(formula, num_permutation, max_id):
    for _ in range(num_permutation):
        # Create a permutation of variable IDs
        var_permute = list(range(1, max_id + 1))
        random.shuffle(var_permute)
        var_permute_dict = {i: var_permute[i - 1] for i in range(1, max_id + 1)}

        # Apply the same permutation to all clauses
        permuted_formula = []
        for clause in formula:
            permuted_clause = [var_permute_dict[abs(var)] * (1 if var > 0 else -1) for var in clause]
            permuted_formula.append(permuted_clause)

        # Shuffle the order of the clauses
        random.shuffle(permuted_formula)

        yield permuted_formula


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('n_samples', action='store', type=int)


    parser.add_argument('output', action='store', type=str)

    parser.add_argument('--min_n', action='store', dest='min_n', type=int, default=15)
    parser.add_argument('--max_n', action='store', dest='max_n', type=int, default=25)

    # Prob of base clause having 1 literals instead of 2
    parser.add_argument('--p_k_2', action='store', dest='p_k_2', type=float, default=1)
    # Geometric distribution parameter for additional number of literals in a clause
    parser.add_argument('--p_geo', action='store', dest='p_geo', type=float, default=1)
    # Expected number of literals per clause: 1 + p_k_2 + (1 / p_geo)
    # Default: Each clause have 3 literals (3-SAT)
    # Alternate combination also with expected value 3: p_k_2=0 p_geo=0.5

    parser.add_argument('--var', action='store_true', help='Use predefined variable clause length distribution instead of 3-SAT. Overrides p_k_2 and p_geo.')
    parser.add_argument('--two_sat', action='store_true', help='Use 2-SAT distribution instead of 3-SAT. Overrides p_k_2 and p_geo. Equivalent to p_k_2=0 p_geo=1')
    parser.add_argument('--skewed', action='store_true', help='Use Assignment First Generation.')
    parser.add_argument('--skew', action='store', type=float, default=0.75, help='Skewness of the assignment distribution. Default: 0.75')

    parser.add_argument('--permute', action='store_true')

    parser.add_argument('--max_id', action='store', dest='max_id', type=int, default=None)
    parser.add_argument('--marginal', action='store_true')

    parser.add_argument('--num_permutation', action='store', dest='num_permutation', type=int, default=5)

    parser.add_argument('--ratio_min', action='store', dest='ratio_min', type=float, default=4.1)
    parser.add_argument('--ratio_max', action='store', dest='ratio_max', type=float, default=4.4)

    parser.add_argument('--test_ratio', action='store_true', help="Test the phase change ratio of the given options")

    parser.add_argument('--seed', action='store', dest='seed', type=int, default=31415)

    opts = parser.parse_args()

    if not opts.permute:
        opts.num_permutation = 1

    random.seed(opts.seed)

    if opts.var:
        print("Overriding p_k_2=0 p_geo=0.5")
        opts.p_k_2 = 0
        opts.p_geo = 0.5

    if opts.two_sat:
        print("Overriding p_k_2=0 p_geo=1")
        opts.p_k_2 = 0
        opts.p_geo = 1
        if opts.ratio_min == 4.1 and opts.ratio_max == 4.4:
            print("Overriding ratio_min=0.8 ratio_max=1.2")
            opts.ratio_min = 0.8
            opts.ratio_max = 1.2

    if opts.test_ratio:
        sat_ratio_list = []
        unsat_ratio_list = []
        for pair in tqdm(range(opts.n_samples // 2)):
            n_vars, sat_base_formula, unsat_base_formula = generate_pair(opts)
            sat_ratio_list.append(len(sat_base_formula) / n_vars)
            unsat_ratio_list.append(len(unsat_base_formula) / n_vars)
        print("SAT ratio: min: %.2f avg: %.2f max: %.2f" % (min(sat_ratio_list), sum(sat_ratio_list) / len(sat_ratio_list), max(sat_ratio_list)))
        print("UNSAT ratio: min: %.2f avg: %.2f max: %.2f" % (min(unsat_ratio_list), sum(unsat_ratio_list) / len(unsat_ratio_list), max(unsat_ratio_list)))
        exit()


    with open(opts.output, 'w') as f:
        # Generate pairs of SAT/UNSAT formulas that differ by one clause
        for pair in tqdm(range(opts.n_samples // 2 // opts.num_permutation)):
            # if opts.marginal:
            #     n_vars, iclauses, iclause_unsat, iclause_sat = gen_marginal_pair(opts)
            #     while len(iclauses) < 1 or (len(iclauses) + 1) / n_vars < opts.ratio_min or (len(iclauses) + 1) / n_vars > opts.ratio_max:
            #         n_vars, iclauses, iclause_unsat, iclause_sat = gen_marginal_pair(opts)
            #     #print("Generated pair %d/%d with clause variable ratio %.2f" % (pair + 1, opts.n_samples // 2 // opts.num_permutation, (len(iclauses) + 1) / n_vars))

            #     sat_base_formula = iclauses + [iclause_sat]
            #     unsat_base_formula = iclauses + [iclause_unsat]
            # elif opts.assign:
            #     n_vars, iclauses, iclause_unsat, iclause_sat = generate_assignment_pair(opts)
            #     while len(iclauses) < 1 or (len(iclauses) + 1) / n_vars < opts.ratio_min or (len(iclauses) + 1) / n_vars > opts.ratio_max:
            #         n_vars, iclauses, iclause_unsat, iclause_sat = generate_assignment_pair(opts)
            #     sat_base_formula = iclauses + [iclause_sat]
            #     unsat_base_formula = iclauses + [iclause_unsat]
            # else:
            #     n_vars, sat_base_formula, unsat_base_formula = gen_random_pair(opts)
                #print("Generated pair %d/%d with clause variable ratio %.2f" % (pair + 1, opts.n_samples // 2 // opts.num_permutation, len(sat_base_formula) / n_vars))
            n_vars, sat_base_formula, unsat_base_formula = generate_pair(opts)
            if opts.permute:
                for sat_formula in permute_formula(sat_base_formula, opts.num_permutation, n_vars if opts.max_id is None else opts.max_id):
                    f.write(ch_form(sat_formula) + "\n")

                for unsat_formula in permute_formula(unsat_base_formula, opts.num_permutation, n_vars if opts.max_id is None else opts.max_id):
                    f.write(ch_form(unsat_formula) + "\n")
            else:
                f.write(ch_form(sat_base_formula) + "\n")
                f.write(ch_form(unsat_base_formula) + "\n")
            
        #f.write(optimal() + "\n")