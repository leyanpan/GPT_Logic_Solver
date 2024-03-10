from dpll import dpll
from AssignTrace import AssignTrace
from heuristics import custom_heuristic

def clause_score(formula, new_clause, heuristic, punishment=-10, n_vars=None):
    # Call DPLL on original formula
    if n_vars is None:
        n_vars = max([abs(lit) for clause in formula for lit in clause])
    trace_orig = AssignTrace()
    assignment = [None] * (n_vars + 1)
    res_orig, assignment = dpll(formula, n_vars, heuristic, assignment, trace_orig)
    num_calls_orig = trace_orig.get_count()
    # Call DPLL on formula with new clause
    trace_new = AssignTrace()
    new_formula = formula + [new_clause]
    assignment = [None] * (n_vars + 1)
    res_new, assignment = dpll(new_formula, n_vars, heuristic, assignment, trace_new)
    num_calls_new = trace_new.get_count()
    if res_orig == 'SAT' and res_new == 'UNSAT':
        return punishment
    return (num_calls_orig / num_calls_new) - 1

if __name__ == '__main__':
    # Load first formula from datasets/SAT_Dataset.txt
    print('Enter dataset file name:')
    file_name = input()
    if file_name == '':
        file_name = 'datasets/SAT_Dataset_Balanced.txt'
    print('Enter line number:')
    line_number = int(input())
    with open(file_name, 'r') as f:
        for i in range(line_number - 1):
            f.readline()
        formula = [[int(lit) for lit in clause.split()] for clause in f.readline().split(' 0 ')[:-1]]

    while True:
        print('Enter new clause:')
        new_clause = [int(lit) for lit in input().split()]
        score = clause_score(formula, new_clause, custom_heuristic)
        print(f'Score: {score}')
    
