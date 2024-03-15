import random

from AssignTrace import AssignTraceClause, AssignTraceDPLL, AssignTraceCDCL
from dpll import dpll, bcp
from cdcl import cdcl
from heuristics import random_heuristic, two_clause_heuristic, custom_heuristic
from utils import translate_trace, parse_dimacs_file, parse_raw_file
# CNF representation Format: List of clauses represented as list of integers, each integer is a variable, negative integers are negated variables
# e.g. [[1, 2, 3], [-1, 2, 3], [1, -2, 3]] is (x1 or x2 or x3) and (not x1 or x2 or x3) and (x1 or not x2 or x3)
# Assignment Format: list of booleans, index i corresponds to variable i, True means variable i is assigned True, False means variable i is assigned False, None means variable i is unassigned


dpll_counter = None
assign_clause = None


if __name__ == '__main__':
    random.seed(0)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input file')
    parser.add_argument('output', nargs='?', help='output file', default=None)
    parser.add_argument('-m', '--heuristic', help='heuristic function', default='custom')
    parser.add_argument('-t', '--trace', help='Execution Trace Type', default='clause')
    parser.add_argument('-f', '--format', default='raw', help='input file format: DIMACS or Raw Clauses')
    parser.add_argument('-a', '--algo', help='Algo to use', default='dpll')
    args = parser.parse_args()
    if args.output is None:
        # Get input file name without path and extension
        import os
        args.output = os.path.splitext(os.path.basename(args.input))[0] + '.out'
        args.output = os.path.join('outputs', args.output)
    dpll_counter = 0
    if args.heuristic == 'random':
        heuristic = random_heuristic
    elif args.heuristic == 'two_clause':
        heuristic = two_clause_heuristic
    elif args.heuristic == 'custom':
        heuristic = custom_heuristic
    else:
        raise ValueError('Invalid heuristic function')
    if args.format == 'dimacs':
        clauses, n_vars, n_clauses = parse_dimacs_file(args.input)
        problems = [(clauses, n_vars)]
    else:
        problems = parse_raw_file(args.input)
    with open(args.output, 'w') as f:
        for clauses, n_vars in problems:
                n_clauses = len(clauses)
                assignment = [None] * (n_vars + 1)
                dpll_counter = 0
                assign_clause = AssignTraceDPLL()
                res = 'UNKNOWN'
                if args.algo == 'cdcl':
                    original_clauses = clauses[:]
                    assign_clause = AssignTraceCDCL(list(clauses))
                    res = cdcl(n_vars, heuristic, assign_clause)
                else:
                    res, new_clauses, assignment = bcp(clauses, assignment, assign_clause)
                    # TODO: should not dpll if bcp is SAT already, doesnt work for 1 0 case
                    res, assignment = dpll(new_clauses, n_vars, heuristic, assignment, assign_clause)
                if args.format == 'dimacs':
                    if res == 'SAT':
                        f.write(f"s cnf 1 {n_vars} {n_clauses}\n")
                        for i in range(1, n_vars + 1):
                            if assignment[i]:
                                f.write(f"v {i}\n")
                            else:
                                f.write(f"v -{i}\n")
                    else:
                        f.write(f"s cnf 0 {n_vars} {n_clauses}\n")
                else:
                    # Write original problem
                    raw_problem = " 0 ".join([" ".join([str(var) for var in clause]) for clause in clauses]) + " 0"
                    if res == 'SAT':
                        assign_clause.sat()
                    elif res == 'UNSAT':
                        assign_clause.unsat()
                    # Use separator between problem and trace
                    f.write(raw_problem + " [SEP] ")
                    if args.trace == 'dpll':
                        f.write(str(assign_clause))
                    elif args.trace == 'clause':
                        f.write(translate_trace(str(assign_clause), res))
                    # Final Solution
                    if res == 'SAT':
                        f.write(f" SAT\n")
                    else:
                        f.write(f" UNSAT\n")