import unittest
from trace_verify import parse_dimacs_trace, verify_rup, extract_assignment_rup, is_valid_assignment, is_formula_satisfied, solve_sat, verify_traces

class TestTraceVerify(unittest.TestCase):

    def setUp(self):
        # Setup method to initialize before each test method
        # Example setup: create a list of clauses for testing

        # SAT correct without backtracking
        self.full_trace_sat_noback = "-2 7 5 0 -5 -3 -6 0 7 2 4 0 -2 -1 -3 0 -5 2 7 0 -7 2 -4 0 5 2 7 0 -7 2 -4 0 7 2 3 0 -1 2 5 0 -3 2 -4 0 4 1 -2 0 1 2 -6 0 -6 -5 2 0 2 4 -7 0 2 -4 7 0 2 7 4 0 2 4 7 0 -4 -2 -5 0 6 4 2 0 7 2 4 0 4 7 2 0 -4 2 -5 0 -2 7 1 0 4 -2 -1 0 7 2 -4 0 -4 2 7 0 4 2 -7 0 4 7 2 0 -6 4 -7 0 [SEP] ( 2 0 ( 1 0 -3 0 4 0 -5 0 7 0 SAT"
        self.trace_1 = "( 2 0 ( 1 0 -3 0 4 0 -5 0 7 0"
        self.formula_1 = [[-2, 7, 5], [-5, -3, -6], [7, 2, 4], [-2, -1, -3], [-5, 2, 7], [-7, 2, -4], [5, 2, 7], [-7, 2, -4], [7, 2, 3], [-1, 2, 5], [-3, 2, -4], [4, 1, -2], [1, 2, -6], [-6, -5, 2], [2, 4, -7], [2, -4, 7], [2, 7, 4], [2, 4, 7], [-4, -2, -5], [6, 4, 2], [7, 2, 4], [4, 7, 2], [-4, 2, -5], [-2, 7, 1], [4, -2, -1], [7, 2, -4], [-4, 2, 7], [4, 2, -7], [4, 7, 2], [-6, 4, -7]]
        self.assignment_1 = [2, 1, -3, 4, -5, 7]
        # SAT correct with backtracking
        self.full_trace_sat_back = "-5 -8 4 0 -5 7 -4 0 7 5 4 0 4 8 -5 0 8 -5 4 0 -5 8 7 0 7 4 -8 0 7 5 8 0 -5 8 4 0 7 -5 8 0 -4 5 8 0 -7 -8 4 0 7 -5 -8 0 -5 -7 -8 0 -4 -8 -5 0 -5 -8 4 0 4 5 8 0 5 4 7 0 8 -4 7 0 -5 7 8 0 8 -5 -7 0 -5 4 7 0 4 5 7 0 7 4 8 0 -5 4 7 0 -4 -5 7 0 -7 -5 8 0 -4 -7 -8 0 7 4 5 0 5 4 7 0 -8 4 -5 0 4 -7 8 0 4 7 5 0 8 5 7 0 [SEP] ( 5 0 ( 8 0 4 0 ) ( -8 0 4 0 7 0 ) ) ( -5 0 ( 7 0 ( 4 0 8 0 ) ( -4 0 -8 0 ) ) ( -7 0 4 0 8 0 SAT"
        self.trace_2 = "( 5 0 ( 8 0 4 0 ) ( -8 0 4 0 7 0 ) ) ( -5 0 ( 7 0 ( 4 0 8 0 ) ( -4 0 -8 0 ) ) ( -7 0 4 0 8 0"
        self.formula_2 = [[-5, -8, 4], [-5, 7, -4], [7, 5, 4], [4, 8, -5], [8, -5, 4], [-5, 8, 7], [7, 4, -8], [7, 5, 8], [-5, 8, 4], [7, -5, 8], [-4, 5, 8], [-7, -8, 4], [7, -5, -8], [-5, -7, -8], [-4, -8, -5], [-5, -8, 4], [4, 5, 8], [5, 4, 7], [8, -4, 7], [-5, 7, 8], [8, -5, -7], [-5, 4, 7], [4, 5, 7], [7, 4, 8], [-5, 4, 7], [-4, -5, 7], [-7, -5, 8], [-4, -7, -8], [7, 4, 5], [5, 4, 7], [-8, 4, -5], [4, -7, 8], [4, 7, 5], [8, 5, 7]]
        self.assignment_2 = [-5, -7, 4, 8]
        # UNSAT correct with backtracking
        self.full_trace_unsat = "6 2 5 0 4 6 9 0 -7 -4 -6 0 5 -7 8 0 -9 3 2 0 7 3 5 0 -9 5 1 0 2 9 -5 0 9 8 -3 0 -7 6 2 0 -9 7 2 0 7 3 -9 0 1 9 3 0 -5 2 1 0 -5 2 -1 0 3 1 -9 0 9 2 7 0 -9 2 5 0 5 4 -2 0 4 6 9 0 -7 -5 -4 0 3 -1 4 0 -4 8 2 0 -7 -5 -2 0 1 2 -9 0 5 -9 -3 0 2 5 -9 0 4 7 2 0 -7 -2 1 0 -2 5 9 0 -9 -4 -7 0 -2 7 -5 0 -7 5 2 0 2 -8 -7 0 3 6 -7 0 -9 5 7 0 -1 -4 -9 0 [SEP] ( 2 0 ( 7 0 -5 0 8 0 4 0 -6 0 1 0 9 0 ) ( -7 0 -5 0 3 0 4 0 -9 0 ) ) ( -2 0 ( 9 0 3 0 7 0 6 0 -4 0 5 0 1 0 ) ( -9 0 -5 0 6 0 7 0 ) ) UNSAT"
        self.trace_3 = "( 2 0 ( 7 0 -5 0 8 0 4 0 -6 0 1 0 9 0 ) ( -7 0 -5 0 3 0 4 0 -9 0 ) ) ( -2 0 ( 9 0 3 0 7 0 6 0 -4 0 5 0 1 0 ) ( -9 0 -5 0 6 0 7 0 ) )"
        self.formula_3 = [[6, 2, 5], [4, 6, 9], [-7, -4, -6], [5, -7, 8], [-9, 3, 2], [7, 3, 5], [-9, 5, 1], [2, 9, -5], [9, 8, -3], [-7, 6, 2], [-9, 7, 2], [7, 3, -9], [1, 9, 3], [-5, 2, 1], [-5, 2, -1], [3, 1, -9], [9, 2, 7], [-9, 2, 5], [5, 4, -2], [4, 6, 9], [-7, -5, -4], [3, -1, 4], [-4, 8, 2], [-7, -5, -2], [1, 2, -9], [5, -9, -3], [2, 5, -9], [4, 7, 2], [-7, -2, 1], [-2, 5, 9], [-9, -4, -7], [-2, 7, -5], [-7, 5, 2], [2, -8, -7], [3, 6, -7], [-9, 5, 7], [-1, -4, -9]]
        self.proof_3 = [[-2, -7], [-2, 7], [-2], [2, -9], [2, 9], [2]]
        # SAT incorrect, actually UNSAT
        self.full_trace_sat_wrong = "-10 -4 3 0 3 -11 1 0 5 -11 -10 0 -1 3 11 0 -8 11 -4 0 -5 -7 1 0 5 3 8 0 7 -10 9 0 2 11 -10 0 -5 7 -8 0 3 1 -7 0 1 -8 -6 0 -2 -3 8 0 -1 2 11 0 -4 2 -3 0 -7 11 -10 0 4 -2 -8 0 -5 -6 -10 0 -1 9 7 0 6 -11 -8 0 -1 -4 9 0 -5 2 -8 0 -5 11 -9 0 -1 5 -2 0 -11 -4 -6 0 1 -4 -7 0 -2 4 8 0 -2 9 -11 0 -1 9 5 0 -6 -3 9 0 -1 10 -8 0 7 -3 6 0 3 10 6 0 -6 9 2 0 7 -1 -2 0 2 7 -5 0 6 -2 1 0 -4 -10 -6 0 6 -10 2 0 1 -5 8 0 -7 9 4 0 -3 -1 -9 0 2 -3 -7 0 -4 11 5 0 4 7 5 0 1 -4 10 0 4 8 -5 0 [SEP] ( 1 0 ( 2 0 5 0 7 0 ( 8 0 4 0 11 0 6 0 ) ( -8 0 -3 0 11 0 9 0 4 0 -6 0 10 0 SAT"
        self.trace_4 = "( 1 0 ( 2 0 5 0 7 0 ( 8 0 4 0 11 0 6 0 ) ( -8 0 -3 0 11 0 9 0 4 0 -6 0 10 0"
        self.formula_4 = [[-10, -4, 3], [3, -11, 1], [5, -11, -10], [-1, 3, 11], [-8, 11, -4], [-5, -7, 1], [5, 3, 8], [7, -10, 9], [2, 11, -10], [-5, 7, -8], [3, 1, -7], [1, -8, -6], [-2, -3, 8], [-1, 2, 11], [-4, 2, -3], [-7, 11, -10], [4, -2, -8], [-5, -6, -10], [-1, 9, 7], [6, -11, -8], [-1, -4, 9], [-5, 2, -8], [-5, 11, -9], [-1, 5, -2], [-11, -4, -6], [1, -4, -7], [-2, 4, 8], [-2, 9, -11], [-1, 9, 5], [-6, -3, 9], [-1, 10, -8], [7, -3, 6], [3, 10, 6], [-6, 9, 2], [7, -1, -2], [2, 7, -5], [6, -2, 1], [-4, -10, -6], [6, -10, 2], [1, -5, 8], [-7, 9, 4], [-3, -1, -9], [2, -3, -7], [-4, 11, 5], [4, 7, 5], [1, -4, 10], [4, 8, -5]]
        self.assignment_4 = [1, 2, 5, 7, -8, -3, 11, 9, 4, -6, 10]
        # SAT malformed, actually UNSAT
        self.full_trace_sat_mal = "10 -5 -7 0 12 -9 -14 0 13 -10 1 0 -10 -6 -12 0 -15 6 -14 0 4 9 -10 0 9 -8 5 0 13 -14 -15 0 -1 -15 13 0 -11 -15 9 0 -9 -13 1 0 7 -4 12 0 -5 -8 -1 0 5 9 3 0 2 -6 10 0 12 9 -7 0 -12 -1 2 0 -4 -1 7 0 -14 -5 3 0 13 -11 -8 0 3 2 9 0 8 5 -2 0 11 3 -4 0 1 -8 -2 0 -2 6 8 0 10 -3 -13 0 -10 -3 11 0 -3 14 7 0 4 -14 -1 0 7 3 1 0 9 -8 3 0 6 -13 11 0 -13 15 -12 0 -3 8 -14 0 11 12 -3 0 15 7 -14 0 5 -8 -15 0 -7 8 13 0 -1 -10 3 0 -2 -8 -4 0 -2 -10 3 0 -9 15 -2 0 12 -6 14 0 7 -14 -10 0 -8 -13 1 0 7 -4 6 0 10 7 -3 0 -11 13 -3 0 -5 14 7 0 9 4 -1 0 5 -6 12 0 -5 1 4 0 -11 5 -13 0 10 1 9 0 4 8 7 0 3 -10 9 0 -4 -7 -15 0 -14 -7 9 0 -4 -1 -8 0 -11 -3 4 0 -3 12 -10 0 6 -9 5 0 12 -6 -14 0 6 8 -12 0 6 15 -10 0 [SEP] ( 3 0 ( 7 0 ( 13 0 10 0 11 0 5 0 ( 1 0 -8 0 4 0 -14 0 ( 12 0 -6 0 2 0 ) ( -1 0 9 0 ) ) ( -1 0 -9 0 4 0 -15 0 -2 0 12 0 -14 0 -8 0 6 0 ) ) ) ) ( -13 0 -10 0 -5 0 -11 0 8 0 9 0 12 0 -2 0 -6 0 -14 0 -15 0 -4 0 ) ) ) ) ( -3 0 ( -3 0 ( 9 0 ( 1 0 -10 0 ( 5 0 -14 0 -7 0 ) ( 1 0 -4 0 7 0 -8 0 13 0 -2 0 -12 0 -6 0 11 0 -15 0 -2 0 9 0 -14 0 -10 0 SAT"
        self.trace_5 = "( 3 0 ( 7 0 ( 13 0 10 0 11 0 5 0 ( 1 0 -8 0 4 0 -14 0 ( 12 0 -6 0 2 0 ) ( -1 0 9 0 ) ) ( -1 0 -9 0 4 0 -15 0 -2 0 12 0 -14 0 -8 0 6 0 ) ) ) ) ( -13 0 -10 0 -5 0 -11 0 8 0 9 0 12 0 -2 0 -6 0 -14 0 -15 0 -4 0 ) ) ) ) ( -3 0 ( -3 0 ( 9 0 ( 1 0 -10 0 ( 5 0 -14 0 -7 0 ) ( 1 0 -4 0 7 0 -8 0 13 0 -2 0 -12 0 -6 0 11 0 -15 0 -2 0 9 0 -14 0 -10 0"
        self.formula_5 = [[10, -5, -7], [12, -9, -14], [13, -10, 1], [-10, -6, -12], [-15, 6, -14], [4, 9, -10], [9, -8, 5], [13, -14, -15], [-1, -15, 13], [-11, -15, 9], [-9, -13, 1], [7, -4, 12], [-5, -8, -1], [5, 9, 3], [2, -6, 10], [12, 9, -7], [-12, -1, 2], [-4, -1, 7], [-14, -5, 3], [13, -11, -8], [3, 2, 9], [8, 5, -2], [11, 3, -4], [1, -8, -2], [-2, 6, 8], [10, -3, -13], [-10, -3, 11], [-3, 14, 7], [4, -14, -1], [7, 3, 1], [9, -8, 3], [6, -13, 11], [-13, 15, -12], [-3, 8, -14], [11, 12, -3], [15, 7, -14], [5, -8, -15], [-7, 8, 13], [-1, -10, 3], [-2, -8, -4], [-2, -10, 3], [-9, 15, -2], [12, -6, 14], [7, -14, -10], [-8, -13, 1], [7, -4, 6], [10, 7, -3], [-11, 13, -3], [-5, 14, 7], [9, 4, -1], [5, -6, 12], [-5, 1, 4], [-11, 5, -13], [10, 1, 9], [4, 8, 7], [3, -10, 9], [-4, -7, -15], [-14, -7, 9], [-4, -1, -8], [-11, -3, 4], [-3, 12, -10], [6, -9, 5], [12, -6, -14], [6, 8, -12], [6, 15, -10]]

        # UNSAT malformed, actually UNSAT
        self.full_trace_unsat_mal = "-1 -7 -8 0 5 1 12 0 9 -13 10 0 12 9 6 0 -9 10 -15 0 11 5 14 0 2 -11 9 0 -4 -13 14 0 9 -11 15 0 5 -7 -8 0 5 -15 14 0 -9 -7 15 0 -11 9 -2 0 8 -7 4 0 -2 1 15 0 15 -11 -8 0 6 -2 9 0 -8 3 2 0 7 1 -8 0 -13 -11 10 0 -3 -11 -8 0 9 15 14 0 15 -6 -9 0 -12 10 6 0 -1 -5 -10 0 -15 7 -11 0 -4 2 -3 0 -15 -9 -1 0 4 1 -13 0 10 5 2 0 -4 -9 -10 0 13 -11 5 0 1 12 10 0 -6 13 -4 0 -1 -11 -8 0 13 14 11 0 4 -8 -14 0 7 5 -6 0 -8 -1 -2 0 12 11 -9 0 8 13 -2 0 -7 1 3 0 -11 15 13 0 14 8 7 0 -14 7 10 0 -2 -3 -10 0 4 -1 -6 0 -14 8 -6 0 10 4 13 0 -2 1 8 0 -13 5 3 0 8 6 -10 0 -7 -11 -4 0 -2 3 -10 0 1 11 -7 0 6 -12 7 0 10 13 4 0 -2 15 -13 0 -7 3 5 0 9 -8 14 0 5 -8 -11 0 5 10 7 0 -8 -10 1 0 13 14 -9 0 [SEP] ( 8 0 ( 1 0 -7 0 -11 0 -2 0 3 0 -4 0 -14 0 5 0 -10 0 -13 0 ) ( -1 0 7 0 5 0 3 0 -11 0 ( 10 0 -2 0 12 0 -4 0 -13 0 -6 0 -14 0 ) ( 9 0 15 0 14 0 -13 0 ) ( ( -9 0 ( -9 0 ( -30 0 ( -30 0 ( -8 0 ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ( -10 0 ( -25 0 ( -1 0 12 0 ( ) ) ( -7 0 -8 0 ( -1 0 ( -25 0 ( -7 0 ( -8 0 14 0 10 0 -6 0 5 0 12 0 9 0 -11 0 -15 0 -2 0 -13 0 -4 0 -3 0 ) ) ) ) ( -14 0 8 0 ( ( -7 0 ( ( -8 0 -7 0 -11 0 14 0 ( ) ) ) ) ) UNSAT"
        self.trace_6 = "( 8 0 ( 1 0 -7 0 -11 0 -2 0 3 0 -4 0 -14 0 5 0 -10 0 -13 0 ) ( -1 0 7 0 5 0 3 0 -11 0 ( 10 0 -2 0 12 0 -4 0 -13 0 -6 0 -14 0 ) ( 9 0 15 0 14 0 -13 0 ) ( ( -9 0 ( -9 0 ( -30 0 ( -30 0 ( -8 0 ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ( ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ( -10 0 ( -25 0 ( -1 0 12 0 ( ) ) ( -7 0 -8 0 ( -1 0 ( -25 0 ( -7 0 ( -8 0 14 0 10 0 -6 0 5 0 12 0 9 0 -11 0 -15 0 -2 0 -13 0 -4 0 -3 0 ) ) ) ) ( -14 0 8 0 ( ( -7 0 ( ( -8 0 -7 0 -11 0 14 0 ( ) ) ) ) )"
        self.formula_6 = [[-1, -7, -8], [5, 1, 12], [9, -13, 10], [12, 9, 6], [-9, 10, -15], [11, 5, 14], [2, -11, 9], [-4, -13, 14], [9, -11, 15], [5, -7, -8], [5, -15, 14], [-9, -7, 15], [-11, 9, -2], [8, -7, 4], [-2, 1, 15], [15, -11, -8], [6, -2, 9], [-8, 3, 2], [7, 1, -8], [-13, -11, 10], [-3, -11, -8], [9, 15, 14], [15, -6, -9], [-12, 10, 6], [-1, -5, -10], [-15, 7, -11], [-4, 2, -3], [-15, -9, -1], [4, 1, -13], [10, 5, 2], [-4, -9, -10], [13, -11, 5], [1, 12, 10], [-6, 13, -4], [-1, -11, -8], [13, 14, 11], [4, -8, -14], [7, 5, -6], [-8, -1, -2], [12, 11, -9], [8, 13, -2], [-7, 1, 3], [-11, 15, 13], [14, 8, 7], [-14, 7, 10], [-2, -3, -10], [4, -1, -6], [-14, 8, -6], [10, 4, 13], [-2, 1, 8], [-13, 5, 3], [8, 6, -10], [-7, -11, -4], [-2, 3, -10], [1, 11, -7], [6, -12, 7], [10, 13, 4], [-2, 15, -13], [-7, 3, 5], [9, -8, 14], [5, -8, -11], [5, 10, 7], [-8, -10, 1], [13, 14, -9]]



    def test_parse_dimacs_trace(self):
        # Test the parse_dimacs_trace method
        formula, trace, res = parse_dimacs_trace(self.full_trace_sat_noback)
        self.assertEqual(formula, self.formula_1)
        self.assertEqual(trace, self.trace_1)
        self.assertEqual(res, "SAT")
        formula, trace, res = parse_dimacs_trace(self.full_trace_sat_back)
        self.assertEqual(formula, self.formula_2)
        self.assertEqual(trace, self.trace_2)
        self.assertEqual(res, "SAT")
        formula, trace, res = parse_dimacs_trace(self.full_trace_unsat)
        self.assertEqual(formula, self.formula_3)
        self.assertEqual(trace, self.trace_3)
        self.assertEqual(res, "UNSAT")
        formula, trace, res = parse_dimacs_trace(self.full_trace_sat_wrong)
        self.assertEqual(formula, self.formula_4)
        self.assertEqual(trace, self.trace_4)
        self.assertEqual(res, "SAT")
        formula, trace, res = parse_dimacs_trace(self.full_trace_sat_mal)
        self.assertEqual(formula, self.formula_5)
        self.assertEqual(trace, self.trace_5)
        self.assertEqual(res, "SAT")
        formula, trace, res = parse_dimacs_trace(self.full_trace_unsat_mal)
        self.assertEqual(formula, self.formula_6)
        self.assertEqual(trace, self.trace_6)
        self.assertEqual(res, "UNSAT")
        
    def test_extract_assignment_rup(self):
        # Test the extract_assignment method
        assignments, _ = extract_assignment_rup(self.trace_1)
        self.assertListEqual(assignments, self.assignment_1)
        assignments, _ = extract_assignment_rup(self.trace_2)
        self.assertListEqual(assignments, self.assignment_2)
        _, rup = extract_assignment_rup(self.trace_3)
        self.assertListEqual(rup, self.proof_3)
        assignments, _ = extract_assignment_rup(self.trace_4)
        self.assertListEqual(assignments, self.assignment_4)
        self.assertTupleEqual(extract_assignment_rup(self.trace_5), (None, None))
        self.assertTupleEqual(extract_assignment_rup(self.trace_6), (None, None))
    
    def test_is_valid_assignment(self):
        # Test the is_valid_assignment method
        self.assertTrue(is_valid_assignment(self.assignment_1))
        self.assertTrue(is_valid_assignment(self.assignment_2))
        self.assertTrue(is_valid_assignment(self.assignment_4))
        self.assertFalse(is_valid_assignment([1, -3, -4, 5, -5]))
        self.assertFalse(is_valid_assignment(None))

    def test_is_formula_satisfied(self):
        # Test the is_formula_satisfied method
        self.assertTrue(is_formula_satisfied(self.formula_1, self.assignment_1))
        self.assertTrue(is_formula_satisfied(self.formula_2, self.assignment_2))
        self.assertFalse(is_formula_satisfied(self.formula_4, self.assignment_4))
        self.assertFalse(is_formula_satisfied(self.formula_5, None))

    def test_rup_proof(self):
        self.assertTrue(verify_rup(self.formula_3, self.proof_3))
        self.assertTrue(verify_rup(self.formula_3, self.proof_3[:-1]))
        self.assertTrue(verify_rup(self.formula_3, self.proof_3[:-2]))
        self.assertFalse(verify_rup(self.formula_3, self.proof_3[:-3]))
        self.assertFalse(verify_rup(self.formula_3, [[2], [9]]))
        self.assertFalse(verify_rup(self.formula_3, None))

    def test_solve_sat(self):
        self.assertEqual(solve_sat(self.formula_1), "SAT")
        self.assertEqual(solve_sat(self.formula_2), "SAT")
        self.assertEqual(solve_sat(self.formula_3), "UNSAT")
        self.assertEqual(solve_sat(self.formula_4), "UNSAT")
        self.assertEqual(solve_sat(self.formula_5), "UNSAT")
        self.assertEqual(solve_sat(self.formula_6), "UNSAT")
    
    def test_verify_traces(self):
        lines = [self.full_trace_sat_noback, self.full_trace_sat_back, self.full_trace_unsat, self.full_trace_sat_wrong, self.full_trace_sat_mal, self.full_trace_unsat_mal]
        correct_sat, correct_unsat, total_sat, total_unsat, total = verify_traces(lines)
        self.assertEqual(correct_sat, 2)
        self.assertEqual(correct_unsat, 1)
        self.assertEqual(total_sat, 2)
        self.assertEqual(total_unsat, 4)
        self.assertEqual(total, 6)


# This allows the test script to be run from the command line
if __name__ == '__main__':
    unittest.main()
