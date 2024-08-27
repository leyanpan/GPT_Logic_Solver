import unittest
import numpy as np
from sops import SOp, ManualArray  # Import relevant classes


class TestSOpOperatorOverrides(unittest.TestCase):
    def setUp(self):
        """
        Set up the test environment, including the error thresholds.
        """
        self.err_thresh = 0.01  # Default error threshold for comparing abstract_eval and concrete_eval
        self.error_margin = 0.2  # Error margin for integer operations with error

    def compare_eval(self, sop: SOp, tokens: list):
        """
        Compare the abstract_eval and concrete_eval results for the given SOp instance.

        Parameters:
        - sop: The SOp instance to test.
        - tokens: The list of tokens to evaluate.

        Asserts that the difference between abstract_eval and concrete_eval is within the error threshold.
        """
        abstract_result = sop.abstract_eval(tokens)
        concrete_result = sop.concrete_eval(tokens)
        diff = np.abs(abstract_result - concrete_result)

        self.assertTrue(
            np.all(diff < self.err_thresh),
            f"Difference between abstract_eval and concrete_eval exceeds threshold: {np.max(diff)}"
        )

    def test_operator_greater_than_without_error(self):
        """
        Test the '>' operator between two SOp instances without considering error tolerance.
        """
        tokens = ["token1", "token2", "token3"]
        arr1 = np.array([[1], [2], [3]])  # Exact integer values
        arr2 = np.array([[0], [2], [4]])
        sop1 = ManualArray(arr1)
        sop2 = ManualArray(arr2)
        result_sop = sop1 > sop2  # Test the '>' operator
        self.compare_eval(result_sop, tokens)

    def test_operator_greater_than_with_error(self):
        """
        Test the '>' operator between two SOp instances considering error tolerance.
        """
        tokens = ["token1", "token2", "token3"]
        arr1 = np.array([[1.1], [2.2], [2.8]])  # Adding error margin to integers
        arr2 = np.array([[0.9], [1.8], [3.2]])
        sop1 = ManualArray(arr1)
        sop2 = ManualArray(arr2)
        result_sop = sop1 > sop2  # Test the '>' operator
        self.compare_eval(result_sop, tokens)

    def test_operator_less_than_without_error(self):
        """
        Test the '<' operator between two SOp instances without considering error tolerance.
        """
        tokens = ["token1", "token2", "token3"]
        arr1 = np.array([[1], [2], [3]])  # Exact integer values
        arr2 = np.array([[2], [3], [4]])
        sop1 = ManualArray(arr1)
        sop2 = ManualArray(arr2)
        result_sop = sop1 < sop2  # Test the '<' operator
        self.compare_eval(result_sop, tokens)

    def test_operator_less_than_with_error(self):
        """
        Test the '<' operator between two SOp instances considering error tolerance.
        """
        tokens = ["token1", "token2", "token3"]
        arr1 = np.array([[1.1], [2.2], [2.8]])  # Adding error margin to integers
        arr2 = np.array([[1.9], [2.8], [3.1]])
        sop1 = ManualArray(arr1)
        sop2 = ManualArray(arr2)
        result_sop = sop1 < sop2  # Test the '<' operator
        self.compare_eval(result_sop, tokens)

    def test_operator_greater_equal_without_error(self):
        """
        Test the '>=' operator between two SOp instances without considering error tolerance.
        """
        tokens = ["token1", "token2", "token3"]
        arr1 = np.array([[1], [2], [3]])  # Exact integer values
        arr2 = np.array([[1], [2], [3]])
        sop1 = ManualArray(arr1)
        sop2 = ManualArray(arr2)
        result_sop = sop1 >= sop2  # Test the '>=' operator
        self.compare_eval(result_sop, tokens)

    def test_operator_greater_equal_with_error(self):
        """
        Test the '>=' operator between two SOp instances considering error tolerance.
        """
        tokens = ["token1", "token2", "token3"]
        arr1 = np.array([[1.2], [2.1], [3.0]])  # Adding error margin to integers
        arr2 = np.array([[1.0], [1.9], [2.8]])
        sop1 = ManualArray(arr1)
        sop2 = ManualArray(arr2)
        result_sop = sop1 >= sop2  # Test the '>=' operator
        self.compare_eval(result_sop, tokens)

    def test_operator_less_equal_without_error(self):
        """
        Test the '<=' operator between two SOp instances without considering error tolerance.
        """
        tokens = ["token1", "token2", "token3"]
        arr1 = np.array([[1], [2], [3]])  # Exact integer values
        arr2 = np.array([[2], [2], [3]])
        sop1 = ManualArray(arr1)
        sop2 = ManualArray(arr2)
        result_sop = sop1 <= sop2  # Test the '<=' operator
        self.compare_eval(result_sop, tokens)

    def test_operator_less_equal_with_error(self):
        """
        Test the '<=' operator between two SOp instances considering error tolerance.
        """
        tokens = ["token1", "token2", "token3"]
        arr1 = np.array([[1.2], [2.1], [3.0]])  # Adding error margin to integers
        arr2 = np.array([[1.4], [2.0], [2.8]])
        sop1 = ManualArray(arr1)
        sop2 = ManualArray(arr2)
        result_sop = sop1 <= sop2  # Test the '<=' operator
        self.compare_eval(result_sop, tokens)

    def test_operator_equal_without_error(self):
        """
        Test the '==' operator between two SOp instances without considering error tolerance.
        """
        tokens = ["token1", "token2", "token3"]
        arr1 = np.array([[1], [2], [3]])  # Exact integer values
        arr2 = np.array([[1], [2], [3]])
        sop1 = ManualArray(arr1)
        sop2 = ManualArray(arr2)
        result_sop = sop1 == sop2  # Test the '==' operator
        self.compare_eval(result_sop, tokens)

    def test_operator_equal_with_error(self):
        """
        Test the '==' operator between two SOp instances considering error tolerance.
        """
        tokens = ["token1", "token2", "token3"]
        arr1 = np.array([[1.1], [2.0], [2.9]])  # Adding error margin to integers
        arr2 = np.array([[0.9], [2.0], [3.1]])
        sop1 = ManualArray(arr1)
        sop2 = ManualArray(arr2)
        result_sop = sop1 == sop2  # Test the '==' operator
        self.compare_eval(result_sop, tokens)

    def test_operator_add_without_error(self):
        """
        Test the '+' operator (addition) between two multi-dimensional SOp instances without considering error tolerance.
        """
        tokens = ["token1", "token2", "token3"]
        arr1 = np.array([[1, 2], [3, 4], [5, 6]])  # 2D array with exact integer values
        arr2 = np.array([[7, 8], [9, 10], [11, 12]])
        sop1 = ManualArray(arr1)
        sop2 = ManualArray(arr2)
        result_sop = sop1 + sop2  # Test the '+' operator
        self.compare_eval(result_sop, tokens)

    def test_operator_add_with_error(self):
        """
        Test the '+' operator (addition) between two multi-dimensional SOp instances considering error tolerance.
        """
        tokens = ["token1", "token2", "token3"]
        arr1 = np.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]])  # 2D array with error margin
        arr2 = np.array([[7.0, 7.9], [8.8, 9.7], [10.6, 11.5]])
        sop1 = ManualArray(arr1)
        sop2 = ManualArray(arr2)
        result_sop = sop1 + sop2  # Test the '+' operator with error
        self.compare_eval(result_sop, tokens)

    def test_operator_sub_without_error(self):
        """
        Test the '-' operator (subtraction) between two multi-dimensional SOp instances without considering error tolerance.
        """
        tokens = ["token1", "token2", "token3"]
        arr1 = np.array([[10, 11], [12, 13], [14, 15]])  # 2D array with exact integer values
        arr2 = np.array([[5, 4], [3, 2], [1, 0]])
        sop1 = ManualArray(arr1)
        sop2 = ManualArray(arr2)
        result_sop = sop1 - sop2  # Test the '-' operator
        self.compare_eval(result_sop, tokens)

    def test_operator_sub_with_error(self):
        """
        Test the '-' operator (subtraction) between two multi-dimensional SOp instances considering error tolerance.
        """
        tokens = ["token1", "token2", "token3"]
        arr1 = np.array([[10.1, 11.2], [12.3, 13.4], [14.5, 15.6]])  # 2D array with error margin
        arr2 = np.array([[4.9, 3.8], [2.7, 1.6], [0.5, -0.4]])
        sop1 = ManualArray(arr1)
        sop2 = ManualArray(arr2)
        result_sop = sop1 - sop2  # Test the '-' operator with error
        self.compare_eval(result_sop, tokens)

    def test_operator_mul_without_error(self):
        """
        Test the '*' operator (multiplication) between two multi-dimensional SOp instances without considering error tolerance.
        """
        tokens = ["token1", "token2", "token3"]
        arr1 = np.array([[2, 3], [4, 5], [6, 7]])  # 2D array with exact integer values
        arr2 = np.array([[1, 2], [3, 4], [5, 6]])
        sop1 = ManualArray(arr1)
        sop2 = ManualArray(arr2)
        result_sop = sop1 * sop2  # Test the '*' operator
        self.compare_eval(result_sop, tokens)

    def test_operator_mul_with_error(self):
        """
        Test the '*' operator (multiplication) between two multi-dimensional SOp instances considering error tolerance.
        """
        tokens = ["token1", "token2", "token3"]
        arr1 = np.array([[2.1, 3.0], [4.2, 5.1], [6.2, 7.3]])  # 2D array with error margin
        arr2 = np.array([[1.0, 2.2], [2.8, 4.0], [4.9, 6.1]])
        sop1 = ManualArray(arr1)
        sop2 = ManualArray(arr2)
        result_sop = sop1 * sop2  # Test the '*' operator with error
        self.compare_eval(result_sop, tokens)




if __name__ == '__main__':
    unittest.main()
