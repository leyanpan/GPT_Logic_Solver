from AssignTrace import AssignTraceCDCL

def bcp(tracer: AssignTraceCDCL):
    # If a clause only has one variable left, assign it
    def get_bad_clause(clauses, var):
        for idx, clause in enumerate(clauses):
            if var in clause:
                continue
            elif -var in clause:
                # Just contains -var
                # See above in update_formula when UNSAT returned
                if len(set(clause)) == 1:
                    return idx
        return None

    while True:
        updated = False
        for idx, clause in enumerate(tracer.clauses):
            if len(clause) == 1:
                var = clause[0]
                tracer.passive_assign(var, tracer.original_clauses[idx])

                if tracer.res == 'UNSAT':
                    # UNSAT means both x and -x were implied
                    # Now find the two clauses that implied x and -x

                    bad_idx = get_bad_clause(tracer.clauses, var)
                    assert(bad_idx is not None)
                    
                    blame_list = tracer.original_clauses[idx] + tracer.original_clauses[bad_idx]
                    tracer.implication_graph[abs(var)] = blame_list
                    learned_clause = analyze_conflict(tracer, blame_list, var)
                    return 'UNSAT', learned_clause
                if tracer.res == 'SAT':
                    return 'SAT', None
                updated = True
                break
        
        if not updated:
            break
    return tracer.res, None

def bcp_polarity(tracer: AssignTraceCDCL):
    # If only one polarity, assign it
    # Polarity assignments are only needed to be done once at the beginning
    # These vars should never be blamed in implication graph
    # TODO: This might help a bit, but is not needed
    # TODO: Doing polarity in loop makes blame graph very slow/complicated?
    # If only one polarity, assign it
    # The actives to blame are those clauses that negation is in
    # vars = set()
    # for clause in tracer.clauses:
    #     for var in clause:
    #         vars.add(var)
    # for var in vars:
    #     if -var not in vars:
    #         updated = True
    #         # Should never be blamed in implication graph
    #         tracer.passive_assign(var, [])
    #         assert tracer.res != 'UNSAT'
    #         if tracer.res == 'SAT':
    #             return 'SAT', None
    # while True:
    #     updated = False
        
    return 'UNKNOWN'

def cdcl(n_vars, heuristic, tracer, MAX_ITER=None):
    # if bcp_polarity(tracer) == 'SAT':
    #     return 'SAT'
    res, learned_clause = bcp(tracer)
    while True:
        if MAX_ITER is not None and tracer.get_count() > MAX_ITER:
            tracer.tokens.append("MAX_ITER")
            return 'UNKNOWN'
        
        while res == 'UNSAT':
            if learned_clause is not None and len(learned_clause) > 0:
                tracer.add_clause(learned_clause)
                backjump(n_vars,learned_clause, tracer)
                res, learned_clause = bcp(tracer)
            else:
                return 'UNSAT'
        
        if res == 'SAT':
            return 'SAT'
        
        # Choose a variable to assign
        rem_vars = [x for x in range(1, n_vars + 1) if tracer.is_assigned(x) is False]
        assert(len(rem_vars) > 0)
        var = heuristic(tracer.clauses, rem_vars)
        tracer.active_assign(var)
        res, learned_clause = bcp(tracer)
            
def backjump(n_vars, learned_clause, tracer: AssignTraceCDCL):
    # Unassign all active and passive variables above highest level in learned_clause
    decision_levels = [tracer.get_decision_level(abs(var)) for var in learned_clause]
    unique_decision_levels = sorted(set(decision_levels), reverse=True)

    # TODO: this is optimal right?
    backjump_level = unique_decision_levels[0] - 1

    for var in range(1, n_vars+1):
        if tracer.get_decision_level(var) > backjump_level:
            tracer.unassign(var)

    return backjump_level


def analyze_conflict(tracer: AssignTraceCDCL, conflict_clause, var):
    # Go down the DAG and find the active variables
    queue = [(conflict_clause, var)]
    learned_clause = set()

    while queue:
        clause, var = queue.pop()
        for cvar in clause:
            if abs(cvar) != abs(var):
                abs_cvar = abs(cvar)
                if abs_cvar in tracer.active_assignment:
                    # Should get opposite of current active polarity
                    learned_clause.add(-tracer.get_polarity(cvar)) 
                else:
                    reason = tracer.implication_graph.get(abs_cvar, None)
                    if reason is not None:
                        queue.append((reason, -cvar))
    return list(learned_clause)