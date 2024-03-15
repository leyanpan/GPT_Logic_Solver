import random
from AssignTrace import AssignTrace, BCPTrace

random.seed(31415)
# Baseline Random heuristic
def random_heuristic(clauses, rem_vars, tracer=None):
    return random.choice(rem_vars) * random.choice([-1, 1])

# Pick the variable that appears in the most 2-clauses
def two_clause_heuristic(clauses, rem_vars, tracer=None):
    vars = {}
    for clause in clauses:
        if len(clause) == 2:
            for var in clause:
                if var not in vars:
                    vars[var] = 0
                vars[var] += 1
    if len(vars) == 0:
        return random_heuristic(clauses, rem_vars)
    return max(vars, key=vars.get)

def custom_heuristic(clauses, rem_vars, tracer=None):
    vars_score = {}
    for clause in clauses:
        for var in clause:
            abs_var = abs(var)
            
            if abs_var not in vars_score:
                vars_score[abs_var] = [0, 0]  # [total_occurrence, positive_bias]
            
            # Increment occurrence
            vars_score[abs_var][0] += 2 ** -(len(clause) - 1)
            # Increment positive bias if var is positive
            vars_score[abs_var][1] += 2 ** -(len(clause) - 1)
    
    if not vars_score:
        # Fallback to random heuristic if vars_score is empty
        return random_heuristic(clauses, rem_vars)
    
    # Select variable based on highest occurrence; tie-breaker: polarity bias
    chosen_var = max(vars_score.keys(), key=lambda x: (vars_score[x][0], vars_score[x][1]))
    # Assign true/false based on the most common polarity
    return chosen_var if vars_score[chosen_var][1] >= vars_score[chosen_var][0]/2 else -chosen_var