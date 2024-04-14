import sys
from pysat.solvers import Glucose4
from dpll import update_formula, bcp
from AssignTrace import AssignTrace

def extract_assignment_rup(sequence):
    stack = [[]]  # Start with an empty list to hold the successful sequence
    rup = []

    sequence = sequence.strip().split()
    for token in sequence:
        if token == '(':
            # Start a new subsequence
            stack.append([])
        elif token == ')':
            # End of a subsequence
            if not stack:
                return None, None
            rup.append([-subseq[0] for subseq in stack[1:] if subseq])
            stack.pop()
        elif (token.isdigit() or token[0] == '-') and token != '0':
            # Parse a number and add to the current subsequence
            try:
                num = int(token)
                stack[-1].append(num)
            except ValueError:
                return None, None

    assignment = []
    for subseq in stack:
        if len(subseq) > 0:
            assignment = assignment + subseq

    # The remaining list in the stack is the successful sequence
    return assignment, rup

def verify_all_steps(formula, trace, res):
    stack = [formula]
    decisions = [(None, [])]
    sequence = trace.strip().split()
    correct_steps = 0
    # Boolean value to keep track of whether next literal is unit propagation or decision
    is_up = True
    # list of unit clauses from last formula update
    unit_clauses = []
    for c in formula:
        if len(c) == 1:
            unit_clauses.append(c[0])
    for token in sequence:
        if token == '(':
            # Start a new subsequence
            stack.append(stack[-1])
            is_up = False
        elif token == ')':
            if not stack:
                return False, correct_steps
            seq = stack.pop()
            decision = decisions.pop()
            if len(decision[1]) == 2 and decision[1][0] == -decision[1][1]:
                continue
            # Only allow backtrack if there is a conflict i.e. empty clause
            if [] not in seq:
                return False, correct_steps
            is_up = True
            correct_steps += 1
        elif (token.isdigit() or token[0] == '-') and token != '0':
            # Parse a number and add to the current subsequence
            try:
                num = int(token)
            except ValueError:
                return False
            # Unit Propagation or Polarity
            if is_up and num not in unit_clauses and any(-num in c for c in stack[-1]):
                return False, correct_steps
            if not is_up:
                # decision literal. Make sure that the literal is in the formula
                if any([abs(num) == abs(d[0]) for d in decisions if d[0]]):
                    return False, correct_steps
                decisions[-1][1].append(num)
                decisions.append((num, []))
            stack[-1], unit_clauses = update_formula(stack[-1], num, True)
            is_up = True
            correct_steps += 1
        elif token == '0':
            pass
        else:
            return False, correct_steps
    if res == 'SAT':
        return len(stack[-1]) == 0, correct_steps
    else:
        return len(stack) == 1, correct_steps

            

def is_valid_assignment(assignment):
    """
    Checks if an assignment is valid, i.e., it does not contain both x and -x for any x.
    """
    if not assignment:
        return False
    return all(-lit not in assignment for lit in assignment)

def is_formula_satisfied(formula, assignment):
    """
    Checks if the given assignment satisfies the formula.
    """
    if not assignment or not formula:
        return False
    # Convert the assignment into a set for faster lookup.
    assignment_set = set(assignment)
    # Check each clause in the formula
    for clause in formula:
        # If none of the literals in the clause is satisfied by the assignment, the formula is not satisfied.
        if not any(lit in assignment_set for lit in clause):
            return False
    # If all clauses are satisfied, the formula is satisfied.
    return True

def check_cnf_satisfiability(formula, assignment):
    """
    Checks if the assignment is valid and if it satisfies the formula.
    """
    # Check if the assignment is valid
    if not is_valid_assignment(assignment):
        return False
    
    # Check if the assignment satisfies the formula
    if is_formula_satisfied(formula, assignment):
        return True
    else:
        return False
    
def parse_dimacs_trace(line):
    # Split the line into formula and trace, ignoring lines that end with "UNSAT"
    if not line.strip().endswith("SAT"):
        return None, None, "INDET"
    
    if line.strip().endswith("UNSAT"):
        res = "UNSAT"
        line = line[:-5].strip().replace('- ', '-')
    else:
        res = "SAT"
        line = line[:-3].strip().replace('- ', '-')
    
    
    # Extract the formula and the trace from the line
    formula_part, trace_part = line.split("[SEP]")
    
    # Parse the formula into a list of lists
    formula = [list(map(int, clause.split())) for clause in formula_part.strip().split(" 0")[:-1]]
    
    # Return the parsed formula and the trace string
    return formula, trace_part.strip(), res

def extract_conflict_clauses(trace):
    """
    Extract conflict clauses from a given string trace.

    :param trace: A string containing conflict clauses in DIMACS format between <CC> and </CC> tags.
    :return: A list of conflict clauses, each represented as a list of integers.
    """
    conflict_clauses = []
    start_tag = "<CC>"
    end_tag = "</CC>"

    start_index = trace.find(start_tag)
    while start_index != -1:
        end_index = trace.find(end_tag, start_index)

        # Extract the clause between <CC> and </CC>, split by space, and convert to integers
        clause_str = trace[start_index + len(start_tag):end_index].strip()
        clause = [int(lit) for lit in clause_str.split()]
        conflict_clauses.append(clause)

        # Find the next occurrence of <CC>
        start_index = trace.find(start_tag, end_index)

    return conflict_clauses


def verify_rup(formula, rup_proof):
    if not rup_proof:
        return False
    formula = formula.copy()
    max_id = max([abs(lit) for clause in formula for lit in clause])
    tracer = AssignTrace()
    assignment = [None] * (max_id + 1)
    res, formula, _ = bcp(formula, assignment, tracer)
    if res == 'UNSAT':
        return True
    for clause in rup_proof:
        cur_formula = formula
        for lit in clause:
            cur_formula = update_formula(cur_formula, -lit)
            if cur_formula == 'UNSAT':
                break
        if cur_formula == 'UNSAT':
            continue
        res, _, _ = bcp(cur_formula, assignment, tracer)
        if res != 'UNSAT':
            return False
        formula += [clause]
    res, _, _ = bcp(formula, assignment, tracer)
    if res == 'UNSAT':
        return True
    return False



def solve_sat(formula):
    solver = Glucose4(bootstrap_with=formula)
    correct_ans = solver.solve()
    correct_ans = "SAT" if correct_ans else "UNSAT"
    return correct_ans

def verify_traces(lines):
    """
    Return the number of SAT formulas with correct assignments and UNSAT formulas with correct RUP proofs.
    """
    correct_pred = 0
    correct_sat = 0
    correct_unsat = 0
    all_correct = 0
    total_sat = 0
    total_unsat = 0
    total = 0
    cdcl = False
    # Read the file and parse the formula and trace
    for i, line in enumerate(lines):
        total += 1
        line = line[:line.index("SAT") + 3]
        if not line.strip().endswith("SAT"):
            continue
        formula, trace, res = parse_dimacs_trace(line)
        correct_ans = solve_sat(formula)
        if correct_ans == "SAT":
            total_sat += 1
        if correct_ans == "UNSAT":
            total_unsat += 1
        if res != correct_ans:
            continue
        correct_pred += 1
        if res == "SAT":
            assert formula is not None
            assert trace is not None
            # Parse the trace into a list of assignments
            assignments, _ = extract_assignment_rup(trace)
            # Check if the formula is satisfied by each assignment
            if check_cnf_satisfiability(formula, assignments):
                correct_sat += 1
            else:
                continue
        if res == "UNSAT":
            assert formula is not None
            assert trace is not None
            # Parse the trace into a list of assignments
            if cdcl:
                conflict_clauses = extract_conflict_clauses(trace)
            else:
                _, conflict_clauses = extract_assignment_rup(trace)
            if verify_rup(formula, conflict_clauses):
                correct_unsat += 1
            else:
                continue
        if verify_all_steps(formula, trace, res)[0]:
            all_correct += 1
    
    return all_correct, correct_pred, correct_sat, correct_unsat, total_sat, total_unsat, total

if __name__ == "__main__":
    # print(extract_conflict_clauses("( -1 ( -20 ( -19 ( -18 9 5 8 -17 15 -16 -4 -3 -2 11 -13 -10 -14 <CC> 18 19 1 </CC> ) 18 11 -12 16 6 -4 -2 -13 -3 -17 -9 7 <CC> 19 1 </CC> ) ) 19 ( -17 18 ( -2 -3 -12 8 15 -4 -5 6 7 -14 <CC> 2 17 -19 </CC> ) 2 -20 13 6 -4 -7 <CC> 17 1 </CC> ) 17 5 <CC> 1 </CC> ) 1 ( 17 ( 2 ( 5 ( 19 7 -8 16 18 10 <CC> -19 -5 </CC> ) -19 ( -8 -18 -9 13 7 <CC> 8 19 -17 </CC> ) 8 -12 -3 -16 6 -11 -4 <CC> -5 -2 -17 </CC> ) -5 -19 11 8 -12 16 14 6 -3 <CC> 5 -17 </CC> ) 5 -19 -2 8 -3 -6 -16 -11 18 7 12 -10 -14 -20 <CC> -17 </CC> ) -17 ( 8 -3 15 -4 ( -19 -13 ( 5 -16 11 18 10 -12 <CC> -5 19 -8 </CC> ) -5 11 -14 7 -12 <CC> 19 -8 </CC> ) 19 7 -5 18 2 16 -6 -14 <CC> -8 </CC> ) -8 -3 ( -5 -14 -19 11 -9 -12 -20 18 10 13 16 15 6 4 7 <CC> 5 </CC> ) 5 -19 -9 18 11 -12 -20 13 16 15 10 6 4 7 -14"))
    # Get the file name from the command line arguments
    if len(sys.argv) < 2:
        print("Usage: python3 trace_verify.py <file_name>")
        sys.exit(1)
    file_name = sys.argv[1]
    with open(file_name, "r") as f:
        lines = f.readlines()
    all_correct, correct_pred, correct_sat, correct_unsat, total_sat, total_unsat, total = verify_traces(lines)
    print(f"Total Fully Correct: {all_correct}/{total} ({all_correct / total * 100:.2f}%)")
    print(f"SAT/UNSAT Correct: {correct_pred}/{total} ({correct_pred / total * 100:.2f}%)")
    print(f"Correct SAT: {correct_sat}/{total_sat} ({correct_sat / total_sat * 100:.2f}%)")
    print(f"Correct UNSAT: {correct_unsat}/{total_unsat} ({correct_unsat / total_unsat * 100:.2f}%)")