import unittest
from AssignTrace import AssignTraceState

class TestAssignTraceState(unittest.TestCase):

    def setUp(self):
        # Setup method to initialize before each test method
        # Example setup: create a list of clauses for testing
        self.clauses = [[1, -2, 3], [-1, 2, -3], [2, 3, -1], [-1, -2, -3]]

    def test_active_assign(self):
        assign_trace_state = AssignTraceState(self.clauses)
        # Test the active_assign method
        assignment = [None, None, None, None]
        var = 2
        assign_trace_state.active_assign(assignment, var)
        print("Testing Active Assign")
        print(assign_trace_state)
        # Check the state and tokens list for expected changes
        self.assertIn("D 2", assign_trace_state.state)
        self.assertIn("Decide 2 |", " ".join(assign_trace_state.tokens))
        self.assertListEqual([None, None, True, None], assign_trace_state.assignment)
        self.assertListEqual([None, None, True, None], assignment)
        # Add more assertions as needed

    def test_passive_assign(self):
        assign_trace_state = AssignTraceState(self.clauses)
        # Test the passive_assign method
        assignment = [None, True, None, True]
        state = ["D 1", "D 3"]
        assign_trace_state.state = state
        var = 2
        assign_trace_state.passive_assign(assignment, var)
        print("Testing Passive Assign")
        print(assign_trace_state)
        # Check for expected UP (Unit Propagation) or Decide tokens
        # Example assertion (adapt based on actual logic and expectations)
        self.assertIn("UP", assign_trace_state.tokens)
        self.assertListEqual([None, True, True, True], assign_trace_state.assignment)
        self.assertListEqual([None, True, True, True], assignment)


    def test_unsat(self):
        assign_trace_state = AssignTraceState(self.clauses)
        # Test the unsat (Unsatisfiable) method
        assignment = [None, True, True, True]
        assign_trace_state.assignment = assignment
        state = ["D 1", "D 3", "2"]
        assign_trace_state.state = state
        assign_trace_state.unsat()
        print("Testing Unsat")
        print(assign_trace_state)
        self.assertIn("BackTrack", assign_trace_state.tokens)
        self.assertListEqual([None, True, None, False], assign_trace_state.assignment)

    def test_simulated_dpll(self):
        assign_trace_state = AssignTraceState(self.clauses)
        # Test the simulated_dpll method
        assignment = [None, None, None, None]
        # Decide 1
        assign_trace_state.active_assign(assignment, 1)
        # Decide 3
        assign_trace_state.active_assign(assignment, 3)
        # Unit Propagation 2
        assign_trace_state.passive_assign(assignment, 2)
        # Conflict
        assign_trace_state.unsat()
        # Decide -3 (As in DPLL implementation, although should be Backtrack)
        assign_trace_state.active_assign(assignment, -3)
        # Unit Propagation 2 by [2, 3, -1]
        assign_trace_state.passive_assign(assignment, 2)
        # This is a SAT assignment
        assign_trace_state.sat()
        print("Testing Simulated DPLL")
        print(assign_trace_state)
        self.assertEqual("| Decide 1 | D 1 | Decide 3 | D 1 D 3 | -1 2 -3 0 UP 2 | D 1 D 3 2 | -1 -2 -3 0 BackTrack -3 | D 1 -3 | 2 3 -1 0 UP 2 | D 1 -3 2 |", str(assign_trace_state).strip())

    def test_full_dpll(self):
        from dpll import dpll
        from heuristics import custom_heuristic
        dimacs = "-6 8 -1 0 1 -6 4 0 -1 -8 -5 0 6 -8 -9 0 -8 6 -4 0 -7 5 -6 0 -2 -4 -9 0 6 8 9 0 -7 9 -5 0 -7 -4 6 0 1 -3 6 0 -8 3 9 0 -6 -1 -3 0 1 9 -4 0 -2 -8 -1 0 2 -6 -3 0 -4 8 -3 0 -1 8 -7 0 4 -3 -2 0 4 -6 -5 0 -8 -6 4 0 5 -4 6 0 -4 7 2 0 6 -9 -3 0 -7 4 -9 0 -9 -8 5 0 -1 -9 4 0 1 6 -7 0 -3 -5 8 0 6 8 -5 0 -5 2 4 0 -6 7 -9 0 3 8 9 0 1 -3 -4 0 6 -3 -8 0 3 -4 -7 0 4 8 1 0"
        elements = dimacs.split()
        clauses = []
        current_clause = []
        num_vars = 9
        assignment = [None] * (num_vars + 1)
        tracer = AssignTraceState(clauses)

        for element in elements:
            if element == '0':
                clauses.append(current_clause)
                current_clause = []
            else:
                current_clause.append(int(element))
        res, new_assignment = dpll(clauses, num_vars, custom_heuristic, assignment, tracer)
        print("Testing Full DPLL")
        print(tracer)




# This allows the test script to be run from the command line
if __name__ == '__main__':
    unittest.main()
