from AssignTrace import AssignTrace, AssignTraceState
# DPLL algorithm with custom heuristic and BCP, active_assignments record assignments made by heuristics as opposed to unit clause inference/BCP
def dpll(clauses, n_vars, heuristic, assignment, tracer=None, MAX_ITER=None, polarity=True):
    if tracer is None:
        # We still need to assign so just initialize a dummy tracer
        tracer = AssignTrace()
    tracer.dpll_start()
    if MAX_ITER is not None and tracer.get_count() > MAX_ITER:
        return 'UNKNOWN', None
    # Get variable assignment at current iteration. positive means True, negative means False
    rem_vars = [x for x in range(1, n_vars + 1) if assignment[x] is None]

    #print(len(rem_vars))
    var = heuristic(clauses, rem_vars)
    assert abs(var) in rem_vars
    # update assignment and formula
    assignment[abs(var)] = (var > 0)
    # active assignment
    tracer.active_assign(assignment, var)
    new_clauses = update_formula(clauses, var)
    if new_clauses == 'UNSAT':
        # contradictory assignment
        tracer.unsat()
    else:
        # unit clause inference
        res, new_clauses, new_assignment = bcp(new_clauses, assignment, tracer, polarity=polarity)
        if res == 'SAT':
            return 'SAT', new_assignment
        if res == 'UNSAT':
            tracer.unsat()
        else:
            # Recursive call
            res, new_assignment = dpll(new_clauses, n_vars, heuristic, new_assignment, tracer, MAX_ITER, polarity=polarity)
            if res == 'SAT':
                return 'SAT', new_assignment
            if res == 'UNKNOWN':
                return 'UNKNOWN', None
            if res == 'UNSAT':
                tracer.unsat()
    tracer.unassign(assignment, var)
    # opposite assignment
    if isinstance(tracer, AssignTraceState):
        assignment[abs(var)] = not (var > 0)
    else:
        tracer.active_assign(assignment, -var)
    new_clauses = update_formula(clauses, -var)
    if new_clauses == 'UNSAT':
        # contradictory assignment
        if not isinstance(tracer, AssignTraceState):
            tracer.unsat()
    else:
        # unit clause inference
        res, new_clauses, new_assignment  = bcp(new_clauses, assignment, tracer, polarity=polarity)
        if res == 'SAT':
            return 'SAT', new_assignment
        if res == 'UNSAT':
            if not isinstance(tracer, AssignTraceState):
                tracer.unsat()
        else:
            # Recursive call
            res, new_assignment = dpll(new_clauses, n_vars, heuristic, new_assignment, tracer, MAX_ITER, polarity=polarity)
            if res == 'SAT':
                return 'SAT', new_assignment
            if res == 'UNKNOWN':
                return 'UNKNOWN', None
            if res == 'UNSAT':
                if not isinstance(tracer, AssignTraceState):
                    tracer.unsat()
    tracer.unassign(assignment, -var)
    return 'UNSAT', None

# Create new formula with var assigned to value
def update_formula(clauses, var):
    new_clauses = []
    for clause in clauses:
        if var in clause:
            continue
        elif -var in clause:
            new_clause = []
            for cvar in clause:
                if cvar == -var:
                    continue
                else:
                    new_clause.append(cvar)
            if len(new_clause) == 0:
                return 'UNSAT'
            new_clauses.append(new_clause)
        else:
            new_clauses.append(clause)
    return new_clauses

# repeat unit clause inference until no more unit clauses
def bcp(clauses, assignment, tracer: AssignTrace, polarity=True):
    new_assignment = assignment.copy()
    while True:
        vars = set()
        updated = False
        for clause in clauses:
            if len(clause) == 1:
                var = clause[0]
                tracer.passive_assign(new_assignment, var)
                clauses = update_formula(clauses, var)
                if clauses == 'UNSAT':
                    return 'UNSAT', None, None
                updated = True
                break
        # polarity assignment
        if polarity:
            for clause in clauses:
                for var in clause:
                    vars.add(var)
            for var in vars:
                if -var not in vars:
                    new_assignment[abs(var)] = (var > 0)
                    tracer.passive_assign(new_assignment, var)
                    clauses = update_formula(clauses, var)
                    if clauses == 'UNSAT':
                        return 'UNSAT', None, None
                    updated = True
                    break
        if not updated:
            break
    if len(clauses) == 0:
        return 'SAT', clauses, new_assignment
    return 'UNKNOWN', clauses, new_assignment

def update_bcp(clauses, assignment, var, tracer: AssignTrace):
    remain_assign = [var]
    new_assignment = assignment.copy()
    while remain_assign != []:
        orig_len = len(remain_assign)
        new_clauses = []
        vars = set()
        for clause in clauses:
            if any([x in remain_assign for x in clause]):
                continue
            elif any([-x in remain_assign for x in clause]):
                new_clause = []
                for cvar in clause:
                    if -cvar in remain_assign:
                        continue
                    else:
                        new_clause.append(cvar)
                if len(new_clause) == 0:
                    return 'UNSAT', None, None
                if len(new_clause) == 1:
                    remain_assign.append(new_clause[0])
                vars.update(new_clause)
                new_clauses.append(new_clause)
            else:
                vars.update(clause)
                new_clauses.append(clause)
        for var in vars:
            if -var not in vars:
                remain_assign.append(var)
        clauses = new_clauses
        remain_assign = remain_assign[orig_len:]
        for var in remain_assign:
            tracer.passive_assign(new_assignment, var)
        if len(clauses) == 0:
            return 'SAT', clauses, new_assignment
    return 'UNKNOWN', new_clauses, new_assignment