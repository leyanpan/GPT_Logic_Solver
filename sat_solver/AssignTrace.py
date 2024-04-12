import numpy as np

def update_formula(clauses, var):
    new_clauses = []
    for clause in clauses:
        if var in clause:
            new_clause = []
            new_clauses.append(new_clause)
        elif -var in clause:
            new_clause = []
            for cvar in clause:
                if cvar == -var:
                    continue
                else:
                    new_clause.append(cvar)
            if len(new_clause) == 0:
                return clauses, 'UNSAT'
            new_clauses.append(new_clause)
        else:
            new_clauses.append(clause)
    done = True
    for clause in new_clauses:
        if len(clause) > 0:
            done = False
            break
    if done:
        return new_clauses, 'SAT'
    return new_clauses, 'UNKNOWN'

class AssignTrace:
    def __init__(self):
        self.counter = 0

    def active_assign(self, assignment, var):
        assignment[abs(var)] = (var > 0)

    def passive_assign(self, assignment, var):
        assignment[abs(var)] = (var > 0)

    def unassign(self, assignment, var):
        assignment[abs(var)] = None

    def sat(self):
        pass

    def dpll_start(self):
        self.counter += 1

    def unsat(self):
        pass

    def __str__(self):
        return ""
    
    def get_count(self):
        self.counter += 1
        return self.counter

class AssignTraceClause(AssignTrace):
    def __init__(self):
        self.new_clauses = []
        self.active_assignment = []
        super().__init__()

    def active_assign(self, assignment, var):
        self.active_assignment.append(var)
        super().active_assign(assignment, var)

    def passive_assign(self, assignment, var):
        self.new_clauses.append([-tvar for tvar in self.active_assignment] + [var])
        super().passive_assign(assignment, var)

    def unassign(self, assignment, var):
        if var in self.active_assignment:
            self.active_assignment.remove(var)
        super().unassign(assignment, var)

    def sat(self):
        self.new_clauses += [[tvar] for tvar in self.active_assignment]
        super().sat()

    def unsat(self):
        self.new_clauses += [[-tvar for tvar in self.active_assignment]]
        super().unsat()

    def __str__(self):
        return " 0 ".join([" ".join([str(var) for var in clause]) for clause in self.new_clauses]) + " 0"

class AssignTraceDPLL(AssignTrace):
    def __init__(self):
        self.tokens = []
        self.parentheses = 0
        super().__init__()

    def active_assign(self, assignment, var):
        self.tokens += ["(", str(var), "0"]
        self.parentheses += 1
        super().active_assign(assignment, var)

    def passive_assign(self, assignment, var):
        self.tokens += [str(var), "0"]
        super().passive_assign(assignment, var)

    def sat(self):
        super().sat()

    def unsat(self):
        if self.parentheses > 0:
            self.tokens += [")"]
            self.parentheses -= 1
        super().unsat()

    def __str__(self):
        return " ".join(self.tokens)


class AssignTraceState(AssignTrace):
    def __init__(self, clauses, proof_clause=False):
        self.tokens = ["|"]
        self.clauses = clauses
        self.state = []
        self.assignment = None
        self.proof_clause = proof_clause
        super().__init__()

    def active_assign(self, assignment, var):
        self.tokens += ["Decide", str(var), "|"]
        self.state.append(f"D {var}")
        self.tokens += self.state + ["|"]
        super().active_assign(assignment, var)
        self.assignment = assignment

    def passive_assign(self, assignment, var):
        up_clause = None
        for clause in self.clauses:
            # Find clause responsible for unit propagation
            if var in clause:
                is_up_clause = True
                for cvar in clause:
                    if cvar != var:
                        if assignment[abs(cvar)] == None or assignment[abs(cvar)] == (cvar > 0):
                            is_up_clause = False
                            break
                if is_up_clause:
                    up_clause = clause
                    break
        if up_clause:
            if self.proof_clause:
                self.tokens +=  [str(cvar) for cvar in up_clause] + ["0", "UP"] + [str(var), "|"]
            else:
                self.tokens += ["UP", str(var), "|"]
        else:
            self.tokens += ["Polarity", str(var), "|"]
        self.state.append(f"{var}")
        self.tokens += self.state + ["|"]
        super().passive_assign(assignment, var)
        self.assignment = assignment

    def sat(self):
        super().sat()

    def unsat(self):
        cont_clause = None
        for clause in self.clauses:
            is_cont_clause = True
            for cvar in clause:
                if self.assignment[abs(cvar)] == None or self.assignment[abs(cvar)] == (cvar > 0):
                    is_cont_clause = False
                    break
            if is_cont_clause:
                cont_clause = clause
                break
        assert cont_clause, "No contradicting clause found"
        # Find last decision in self.state and remove all unit propagations after that
        bvar = None
        for i in range(len(self.state) - 1, -1, -1):
            if self.state[i][0] == "D":
                bvar = int(self.state[i][2:])
                break
        if self.proof_clause:
            self.tokens += [str(cvar) for cvar in cont_clause] + ["0"]
        if bvar is None:
            self.tokens += ["BackTrack", "0", "|"]
        else:
            self.tokens += ["BackTrack", str(-bvar), "|"]
            remove = [abs(int(v)) for v in self.state[i + 1:]]
            for r in remove:
                self.assignment[r] = None
            self.state = self.state[:i] + [str(-bvar)]
            self.assignment[abs(bvar)] = not (bvar > 0)
            self.tokens += self.state + ["|"]
        super().unsat()

    def __str__(self):
        return " ".join(self.tokens)

class AssignTraceCDCL(AssignTrace):
    def __init__(self, clauses):
        self.tokens = []
        self.parentheses = 0
        self.active_assignment = []
        self.implication_graph = {} # var -> reason clause
        self.decision_levels = {}
        self.current_level = 0
        self.clauses = clauses[:]
        self.res = None
        self.mapping = {}
        self.assigned = {} # if literal is assigned
        self.original_clauses = clauses[:]
        for i, clause in enumerate(clauses):
            for var in clause:
                if var not in self.mapping:
                    self.mapping[var] = []
                self.mapping[var].append(i)
        super().__init__()
        
    def add_clause(self, clause):
        self.clauses.append(clause)
        self.tokens += ["<CC>"] # Conflict Clause
        for var in clause:
            self.tokens += [str(var)]
        self.tokens += ["</CC>"] # End Conflict Clause
        self.original_clauses.append(clause)
        for var in clause:
            if var not in self.mapping:
                # only runs if bcp_polarity is not turned on
                self.mapping[var] = []
            self.mapping[var].append(len(self.clauses) - 1)
        
        for var in clause:
            if var in self.assigned and self.assigned[var]:
                self.clauses[-1] = []
                break
    
    def active_assign(self, var):
        assert(not self.is_assigned(var))
        self.tokens += ["(", str(var)]
        self.parentheses += 1
        self.active_assignment.append(abs(var))
        self.current_level += 1
        self.decision_levels[abs(var)] = self.current_level
        self.clauses, self.res = update_formula(self.clauses, var)
        self.assigned[var] = True
        self.assigned[-var] = False

    def is_assigned(self, var):
        if var in self.assigned or -var in self.assigned:
            return self.assigned[var] or self.assigned[-var]
        return False

    def passive_assign(self, var, reason_clause):
        assert(not self.is_assigned(var))
        self.tokens += [str(var)]
        self.implication_graph[abs(var)] = reason_clause
        self.decision_levels[abs(var)] = self.current_level
        self.clauses, self.res = update_formula(self.clauses, var)
        self.assigned[var] = True
        self.assigned[-var] = False

    def unassign(self, var):
        if abs(var) in self.active_assignment:
            self.active_assignment.remove(abs(var))
            self.current_level -= 1
            self.tokens += [")"]
            self.parentheses -= 1
        if var in self.assigned or -var in self.assigned:
            self.assigned[var] = False
            self.assigned[-var] = False
            if abs(var) in self.decision_levels:
                del self.decision_levels[abs(var)]
            if abs(var) in self.implication_graph:
                del self.implication_graph[abs(var)]
            
            # repopulate clauses
            total = self.mapping.get(var, []) + self.mapping.get(-var, [])
            for idx in total:
                is_sat = False
                new_clause = []
                for cvar in self.original_clauses[idx]:
                    if cvar in self.assigned and self.assigned[cvar]:
                        is_sat = True
                        break
                    if -cvar in self.assigned and self.assigned[-cvar]:
                        continue
                    else:
                        new_clause.append(cvar)
                if not is_sat:
                    self.clauses[idx] = new_clause
        else:
            assert(False)

    def get_polarity(self, var):
        cvar = abs(var)
        if cvar in self.assigned and self.assigned[cvar]:
            return cvar
        else:
            assert -cvar in self.assigned and self.assigned[-cvar]
            return -cvar
        
    def get_decision_level(self, var):
        return self.decision_levels.get(abs(var), -1)
    
    def sat(self):
        self.tokens += [")"] * self.parentheses
        super().sat()

    def unsat(self):
        if self.parentheses > 0:
            self.tokens += [")"]
            self.parentheses -= 1
        super().unsat()

    def __str__(self):
        return " ".join(self.tokens)

class BCPTrace(AssignTrace):
    def __init__(self, num_vars):
        self.num_vars = num_vars
        self.bcp_est = np.zeros(2 * num_vars + 1)
        super().__init__()

    def heuristic(self, rem_vars, clauses):
        new_est = self.bcp_est.copy()
        for clause in clauses:
            if len(clause) == 2:
                new_est[self.index(clause[0])] += 1
                new_est[self.index(clause[1])] += 1
        self.bcp_est = new_est / 2
        return rem_vars[np.argmax(self.bcp_est[np.array(rem_vars) + self.num_vars])]

    def index(self, var):
        return var + self.num_vars
    
    def var(self, i):
        return i - self.num_vars
