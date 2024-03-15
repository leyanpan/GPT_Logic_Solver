import sys
from pysat.solvers import Glucose3

def find_successful_sequence(sequence):
    stack = [[]]  # Start with an empty list to hold the successful sequence

    # Helper function to parse numbers and control characters in the sequence
    def parse_number(i):
        num = ''
        while i < len(sequence) and (sequence[i].isdigit() or sequence[i] in '-<>/'):
            num += sequence[i]
            i += 1
        return num.strip(), i
    sequence = sequence.strip(') ')
    i = 0
    while i < len(sequence):
        if sequence[i] == '(':
            # Start a new subsequence
            stack.append([])
        elif sequence[i] == ')':
            # End of a subsequence
            subseq = stack.pop()
        elif sequence[i].isdigit() or sequence[i] == '-':
            # Parse a number and add to the current subsequence
            num, i = parse_number(i)
            stack[-1].extend([int(n) for n in num.split() if n.isdigit() or n[1:].isdigit()])
            continue  # Skip the increment step as it's done in parse_number
        i += 1


    ret = []
    for subseq in stack:
        if len(subseq) > 0:
            ret = ret + subseq

    # The remaining list in the stack is the successful sequence
    return ret

def is_valid_assignment(assignment):
    """
    Checks if an assignment is valid, i.e., it does not contain both x and -x for any x.
    """
    return all(-lit not in assignment for lit in assignment)

def is_formula_satisfied(formula, assignment):
    """
    Checks if the given assignment satisfies the formula.
    """
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
    else:
        res = "SAT"
    
    line = line[:-4].replace('- ', '-')
    
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
    g = Glucose3(bootstrap_with=formula)
    for clause in rup_proof:
        res, _ = g.propagate(assumptions=[-lit for lit in clause])
        if res:
            return False
        g.add_clause(clause)
    res, _ = g.propagate()
    if res:
        return False
    return True


if __name__ == "__main__":
    # print(extract_conflict_clauses("( -1 ( -20 ( -19 ( -18 9 5 8 -17 15 -16 -4 -3 -2 11 -13 -10 -14 <CC> 18 19 1 </CC> ) 18 11 -12 16 6 -4 -2 -13 -3 -17 -9 7 <CC> 19 1 </CC> ) ) 19 ( -17 18 ( -2 -3 -12 8 15 -4 -5 6 7 -14 <CC> 2 17 -19 </CC> ) 2 -20 13 6 -4 -7 <CC> 17 1 </CC> ) 17 5 <CC> 1 </CC> ) 1 ( 17 ( 2 ( 5 ( 19 7 -8 16 18 10 <CC> -19 -5 </CC> ) -19 ( -8 -18 -9 13 7 <CC> 8 19 -17 </CC> ) 8 -12 -3 -16 6 -11 -4 <CC> -5 -2 -17 </CC> ) -5 -19 11 8 -12 16 14 6 -3 <CC> 5 -17 </CC> ) 5 -19 -2 8 -3 -6 -16 -11 18 7 12 -10 -14 -20 <CC> -17 </CC> ) -17 ( 8 -3 15 -4 ( -19 -13 ( 5 -16 11 18 10 -12 <CC> -5 19 -8 </CC> ) -5 11 -14 7 -12 <CC> 19 -8 </CC> ) 19 7 -5 18 2 16 -6 -14 <CC> -8 </CC> ) -8 -3 ( -5 -14 -19 11 -9 -12 -20 18 10 13 16 15 6 4 7 <CC> 5 </CC> ) 5 -19 -9 18 11 -12 -20 13 16 15 10 6 4 7 -14"))
    # Get the file name from the command line arguments
    if len(sys.argv) < 2:
        print("Usage: python3 trace_verify.py <file_name>")
        sys.exit(1)
    file_name = sys.argv[1]
    correct_sat = 0
    correct_unsat = 0
    total_sat = 0
    total_unsat = 0
    total = 0
    # Read the file and parse the formula and trace
    with open(file_name, "r") as f:
        for line in f:
            line = line[:line.index("SAT") + 3]
            if not line.strip().endswith("SAT"):
                total += 1
                continue
            formula, trace, res = parse_dimacs_trace(line)
            if res == "SAT":
                assert formula is not None
                assert trace is not None
                # Parse the trace into a list of assignments
                assignments = find_successful_sequence(trace)
                total_sat += 1
                # Check if the formula is satisfied by each assignment
                if check_cnf_satisfiability(formula, assignments):
                    correct_sat += 1
            if res == "UNSAT":
                assert formula is not None
                assert trace is not None
                # Parse the trace into a list of assignments
                conflict_clauses = extract_conflict_clauses(trace)
                total_unsat += 1
                if verify_rup(formula, conflict_clauses):
                    correct_unsat += 1
                

            total += 1

    print(f"Correct: {correct_sat + correct_unsat}/{total} ({(correct_sat + correct_unsat) / total * 100:.2f}%)")
    print(f"Correct SAT: {correct_sat}/{total_sat} ({correct_sat / total_sat * 100:.2f}%)")
    print(f"Correct UNSAT: {correct_unsat}/{total_unsat} ({correct_unsat / total_unsat * 100:.2f}%)")