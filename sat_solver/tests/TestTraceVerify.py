import unittest
from trace_verify import parse_dimacs_trace, verify_rup, extract_assignment

class TestTraceVerify(unittest.TestCase):

    def setUp(self):
        # Setup method to initialize before each test method
        # Example setup: create a list of clauses for testing
        self.full_trace_1 = "-2 7 5 0 -5 -3 -6 0 7 2 4 0 -2 -1 -3 0 -5 2 7 0 -7 2 -4 0 5 2 7 0 -7 2 -4 0 7 2 3 0 -1 2 5 0 -3 2 -4 0 4 1 -2 0 1 2 -6 0 -6 -5 2 0 2 4 -7 0 2 -4 7 0 2 7 4 0 2 4 7 0 -4 -2 -5 0 6 4 2 0 7 2 4 0 4 7 2 0 -4 2 -5 0 -2 7 1 0 4 -2 -1 0 7 2 -4 0 -4 2 7 0 4 2 -7 0 4 7 2 0 -6 4 -7 0 [SEP] ( 2 0 ( 1 0 -3 0 4 0 -5 0 7 0 SAT"
        self.trace_1 = "( 2 0 ( 1 0 -3 0 4 0 -5 0 7 0"
        self.formula_1 = [[-2, 7, 5], [-5, -3, -6], [7, 2, 4], [-2, -1, -3], [-5, 2, 7], [-7, 2, -4], [5, 2, 7], [-7, 2, -4], [7, 2, 3], [-1, 2, 5], [-3, 2, -4], [4, 1, -2], [1, 2, -6], [-6, -5, 2], [2, 4, -7], [2, -4, 7], [2, 7, 4], [2, 4, 7], [-4, -2, -5], [6, 4, 2], [7, 2, 4], [4, 7, 2], [-4, 2, -5], [-2, 7, 1], [4, -2, -1], [7, 2, -4], [-4, 2, 7], [4, 2, -7], [4, 7, 2], [-6, 4, -7]]
        self.assignment_1 = [2, 1, -3, 4, -5, 7]

    def test_parse_dimacs_trace(self):
        # Test the parse_dimacs_trace method
        formula, trace, res = parse_dimacs_trace(self.full_trace_1)
        self.assertEqual(formula, self.formula_1)
        self.assertEqual(trace, self.trace_1)
        self.assertEqual(res, "SAT")
        
    def test_extract_assignment(self):
        # Test the extract_assignment method
        assignments = extract_assignment(self.trace_1)
        self.assertListEqual(assignments, self.assignment_1)


# This allows the test script to be run from the command line
if __name__ == '__main__':
    unittest.main()
