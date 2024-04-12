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
        dimacs = "-9 4 -10 0 -3 9 8 0 9 -3 -10 0 -7 -3 8 0 -2 -8 -5 0 8 -2 -5 0 3 -9 -10 0 6 2 10 0 -3 -9 5 0 -7 8 -5 0 8 -4 6 0 -4 -9 6 0 3 5 -1 0 3 -7 9 0 -9 -7 -8 0 3 -4 8 0 -3 -5 -6 0 4 9 7 0 7 2 -8 0 -9 7 -5 0 6 -9 2 0 2 7 5 0 -7 6 2 0 10 -4 -5 0 1 -7 -5 0 6 9 -7 0 7 -1 8 0 -2 10 -3 0 -5 1 10 0 -5 -7 1 0 4 -3 -10 0 -5 2 -10 0 1 6 3 0 4 8 7 0 -3 6 4 0 -1 -2 -8 0 3 -6 5 0 -9 10 -1 0 3 7 1 0 10 -2 8 0 7 4 -9 0 2 9 -8 0 "
        elements = dimacs.split()
        clauses = []
        current_clause = []
        num_vars = 10
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
