# Converts the trace portion each line in a SAT dataset file
# Original Trace Format: ( decision propogation propogation ... ( decision propogation propogation ... <CC> conflict clause </CC>) )
# New Trace Format: Traslate all progations to clauses using implication
# (i.e. the negation of the decision variables and the propogation variable form a clause)
# Format: propogation clause 1 0 propogation clause 2 0 ...  0 conflict clause 0
# Also use suffix unit clauses to denote the final assignments if the formula is SAT

import sys

def parse_line(line):
    line = line.strip()
    if line.endswith("UNSAT"):
        res = "UNSAT"
        line = line[:-5].strip()
    elif line.endswith("SAT"):
        res = "SAT"
        line = line[:-3].strip()
    else:
        raise ValueError(f"Unknown result {line}")

    line = line.replace('- ', '-')

    # Extract the formula and the trace from the line
    formula_part, trace_part = line.split("[SEP]")

    # Return the parsed formula and the trace string
    return formula_part, trace_part.strip(), res

def translate_trace(trace, res):
    stack = [[]]
    trace_clauses = []
    is_decision = False
    tokens = trace.split()
    final_part = len(tokens)
    while tokens[final_part - 1] == ')':
        final_part -= 1
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == '(':
            is_decision = True
        elif token.startswith('-') or token[0].isdigit():
            if is_decision:
                stack.append([int(token)])
                is_decision = False
            else:
                stack[-1].append(int(token))
                new_clause = [-assign[0] for assign in stack[1:]] + [int(token)]
                if new_clause not in trace_clauses:
                    trace_clauses.append(new_clause)
        elif token == ')':
            cur_assigns = stack.pop()
            if i >= final_part and res == 'SAT':
                for cur_assign in cur_assigns:
                    new_clause = [cur_assign]
                    if new_clause not in trace_clauses:
                        trace_clauses.append(new_clause)
        elif token == '<CC>':
            i += 1
            conflict_clause = []
            while tokens[i] != '</CC>':
                conflict_clause.append(int(tokens[i]))
                i += 1
            trace_clauses.append(conflict_clause)
        else:
            raise ValueError(f"Unknown token {token}")
        i += 1

    return ' 0 '.join(' '.join(map(str, clause)) for clause in trace_clauses) + ' 0'


# Parse DIMACS CNF file
def parse_dimacs_file(fn):
    with open(fn) as f:
        for line in f:
            if line[0] == 'p':
                _, _, n_vars, n_clauses = line.split()
                n_vars, n_clauses = int(n_vars), int(n_clauses)
                break
            elif line[0] == 'c':
                continue
        clauses = []
        cur_clause = []
        for line in f:
            vars = [int(x) for x in line.strip().split()]
            for var in vars:
                if var == 0:
                    # Empty clause directly UNSAT
                    clauses.append(cur_clause)
                    cur_clause = []
                else:
                    cur_clause.append(var)
        if len(cur_clause) > 0:
            clauses.append(cur_clause)
    return clauses, n_vars, n_clauses

def parse_raw_file(fn):
    problems = []
    with open(fn) as f:
        for line in f:
            if '[SEP]' in line:
                line = line[:line.index('[SEP]')]
            num_vars = 0
            clauses = []
            cur_clause = []
            nums = line.split()
            for num in nums:
                if num == '0':
                    clauses.append(cur_clause)
                    cur_clause = []
                else:
                    cur_clause.append(int(num))
                    num_vars = max(num_vars, abs(int(num)))
            if len(cur_clause) > 0:
                clauses.append(cur_clause)
            problems.append((clauses, num_vars))
    return problems

# if __name__ == "__main__":
#     # Get the input file path
#     input_file_path = sys.argv[1]

#     # Get the output file path
#     output_file_path = sys.argv[2]

#     # Open the input file
#     with open(input_file_path, 'r') as input_file:
#         # Open the output file
#         with open(output_file_path, 'w') as output_file:
#             # Iterate through each line in the input file
#             for line in input_file:
#                 # Parse the line
#                 formula, trace, res = parse_line(line)

#                 # Translate the trace
#                 trace = translate_trace(trace, res)

#                 # Write the line to the output file
#                 output_file.write(f"{formula}[SEP] {trace} {res}\n")

