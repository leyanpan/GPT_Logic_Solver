import sys
from pysat.solvers import Glucose4
from dpll import update_formula, bcp
from AssignTrace import AssignTraceState

class TraceAbstract:
    def __init__(self, raw_trace):
        # SAT is action, UNSAT is state (currently -- should fix); action
        # BackTrack 0 amounts to deciding UNSAT, so just replace for now:
        raw_trace = raw_trace.replace(" BackTrack 0 | UNSAT", " UNSAT")

        parts = raw_trace.split("[SEP]")
        self.raw_formula, trace = [x.strip() for x in parts]

        self.formula = [list(map(int, clause.split())) for clause in self.raw_formula.strip().split(" 0")[:-1]]

        steps = trace.split("|")

        self.states = [x.strip() for x in steps[::2]]
        self.actions = [x.strip() for x in steps[1::2]]

        self.solution = self.actions[-1]

    def parse_action(self, action):
        """
        Parse the action string into a tuple of the form (action, arg)
        """
        if action in ["SAT", "UNSAT"]:
            return action, None
        else:
            return action.split(" ")

    def to_prepend(self, curr_state):
        """
        Small helper to see if we need to prepend a space to the state update
        """
        if curr_state == "":
            return ""
        else:
            return " "

    def negate(self, symbol):
        """
        Negate a symbol (for handling backtrack arguments)
        """
        if symbol[0] == "-":
            return symbol[1:]
        else:
            return "-" + symbol


    ### These methods apply a given action to a given state, and return the 
    ### expected resultant state; they're used for local verification of state 
    ### transitions.

    def Decide(self, curr_state, arg):
        return curr_state + self.to_prepend(curr_state) + f"D {arg}"
    
    def UP(self, curr_state, arg):
        return curr_state + self.to_prepend(curr_state) + f"{arg}"

    def BackTrack(self, curr_state, arg):
        decision_literal = self.negate(arg)
        backtrack_decision = "D " + decision_literal + " "

        pos = curr_state.find(backtrack_decision)

        if pos == -1:
            return "Failed to backtrack: decision not found in state."
        else:
            return curr_state[:pos] + arg

    ### -----------------------------------------------------------------------
    
    def get_assignment(self):
        """
        Get the assignment from the last state
        """
        state = self.states[-1].replace("D ", "")
        return [int(lit) for lit in state.split()]

    def is_valid_assignment(self):
        """
        Checks if an assignment is valid, i.e., it does not contain both x and -x for any x.
        """
        assignment = self.get_assignment()

        if not assignment:
            return False
        return all(-lit not in assignment for lit in assignment)

    def is_formula_satisfied(self):
        """
        Checks if the given assignment satisfies the formula.
        """
        assignment = self.get_assignment()
        formula = self.formula

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

    def solve_sat(self):
        """
        Solve the formula using a SAT solver.
        """
        solver = Glucose4(bootstrap_with=self.formula)
        correct_ans = solver.solve()
        solver.delete()

        correct_ans = "SAT" if correct_ans else "UNSAT"
        return correct_ans

    def oll_korrekt(self, verbose=False):
        """
        Iterate through oll state transitions, and (locally) verify their 
        korrektness.  If any are unkorrekt, return false, else return true.
        """

        transitions = self.actions[:-1]
        
        for i in range(len(transitions)):
            action, arg = self.parse_action(transitions[i])

            cur_state = self.states[i]
            next_state_encountered = self.states[i+1]
            next_state_expected = eval(f"self.{action}(cur_state, arg)")

            if next_state_expected != next_state_encountered:
                if verbose:  # for debugging
                    print(f"Current state: {cur_state}")
                    print(f"Transition {i}: {action} {arg}")
                    print(f"Expected state: {next_state_expected}")
                    print(f"Encountered state: {next_state_encountered}")

                return False
        
        return True

def verify_traces(lines):
    correct_pred = 0  # number of correct predictions
    correct_sat = 0  # number of correct SAT predictions
    correct_unsat = 0  # number of correct UNSAT predictions
    all_correct = 0  # number of correct predictions with O.K. state transitions
    total_sat = 0  # number of satisfiable problems
    total_unsat = 0  # number of unsatisfiable problems
    total = 0  # total number of problems

    for i, line in enumerate(lines):
        trace = TraceAbstract(line)
        total += 1

        correct_ans = trace.solve_sat()

        if correct_ans == "SAT":
            total_sat += 1
        if correct_ans == "UNSAT":
            total_unsat += 1

        if trace.solution != correct_ans:
            continue

        correct_pred += 1

        if correct_ans == "SAT":
            correct_sat += 1
        if correct_ans == "UNSAT":
            correct_unsat += 1
        
        if trace.oll_korrekt():
            all_correct += 1

    return correct_pred, correct_sat, correct_unsat, all_correct, total_sat, total_unsat, total


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 trace_verify_state.py <trace_file>")
        sys.exit(1)

    file_name = sys.argv[1]

    with open(file_name, "r") as f:
        lines = f.readlines()

    correct_pred, correct_sat, correct_unsat, all_correct, total_sat, total_unsat, total = verify_traces(lines)

    print(f"Total Fully Correct: {all_correct}/{total} ({all_correct / total * 100:.2f}%)")
    print(f"SAT/UNSAT Correct: {correct_pred}/{total} ({correct_pred / total * 100:.2f}%)")
    print(f"Correct SAT: {correct_sat}/{total_sat} ({correct_sat / total_sat * 100:.2f}%)")
    print(f"Correct UNSAT: {correct_unsat}/{total_unsat} ({correct_unsat / total_unsat * 100:.2f}%)")