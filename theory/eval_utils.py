from sop_utils import sop_generate
from typing import List


def evaluate_model(dpll_model, samples, tokens, dpll_sop, args, print_results=False):
    """
    Evaluate the DPLL model on the given samples and compute statistics.

    Args:
        dpll_model: The compiled DPLL model.
        samples (List[str]): List of sample strings to evaluate.
        tokens: The token set used by the model.
        dpll_sop: The sum-of-products representation of the DPLL model.
        args: Command-line arguments containing evaluation options.

    Returns:
        Tuple containing:
            - accuracy (float): The accuracy of the model on the samples.
            - max_clauses (int): Maximum number of clauses in any prompt.
            - max_cot_len (int): Maximum chain-of-thought length.
            - max_bt (int): Maximum number of backtracking steps.
            - avg_cot_len (float): Average chain-of-thought length.
            - avg_bt (float): Average number of backtracking steps.
    """
    correct_predictions = 0
    total_samples = len(samples)

    max_clauses = 0  # Max number of clauses (number of '0' tokens in the prompt)
    max_cot_len = 0  # Max length of Chain-of-Thought (len difference between prompt and generated tokens)
    max_bt = 0  # Maximum number of backtracking (number of generated '[BT]' tokens)
    total_cot_len = 0  # Total length of CoT for averaging
    total_bt = 0  # Total number of backtracking steps for averaging

    for sample in samples:
        prompt_tokens = sample.strip().split()

        # Determine the ground truth label before removing tokens after [SEP]
        ground_truth_label = None
        if "SAT" in prompt_tokens:
            ground_truth_label = "SAT"
        elif "UNSAT" in prompt_tokens:
            ground_truth_label = "UNSAT"

        # Truncate at [SEP] if present
        if "[SEP]" in prompt_tokens:
            prompt_tokens = prompt_tokens[:prompt_tokens.index("[SEP]") + 1]
        else:
            prompt_tokens += ["[SEP]"]

        if prompt_tokens[0] != "[BOS]":
            prompt_tokens.insert(0, "[BOS]")

        # Update max_clauses (count of '0' in prompt_tokens)
        num_clauses_in_prompt = prompt_tokens.count('0')
        if num_clauses_in_prompt > max_clauses:
            max_clauses = num_clauses_in_prompt

        # Generate completion using the appropriate method
        if args.abstract:
            completion_tokens = sop_generate(prompt_tokens, tokens, dpll_sop, max_generation_length=args.context_len - len(prompt_tokens) - 1)
        elif args.state:
            completion_tokens = dpll_model.state_generate(prompt_tokens)
        else:
            completion_tokens = dpll_model.generate(prompt_tokens)
        if print_results:
            print(" ".join(completion_tokens))

        # Update CoT length statistics
        cot_len = len(completion_tokens) - len(prompt_tokens)
        total_cot_len += cot_len
        if cot_len > max_cot_len:
            max_cot_len = cot_len

        # Update backtracking statistics
        num_bt_tokens = completion_tokens.count('[BT]')
        total_bt += num_bt_tokens
        if num_bt_tokens > max_bt:
            max_bt = num_bt_tokens

        # Determine predicted label from completion
        predicted_label = None
        if "SAT" in completion_tokens:
            predicted_label = "SAT"
        elif "UNSAT" in completion_tokens:
            predicted_label = "UNSAT"

        # Compare predicted label with ground truth label
        if predicted_label == ground_truth_label:
            correct_predictions += 1

    # Calculate accuracy
    accuracy = correct_predictions / total_samples

    # Calculate averages
    avg_cot_len = total_cot_len / total_samples
    avg_bt = total_bt / total_samples

    # Return the statistics
    return accuracy, max_clauses, max_cot_len, max_bt, avg_cot_len, avg_bt

def infer_args_from_file(samples: List[str]):
    num_vars = 0
    num_clauses = 0
    context_len = 0

    for line in samples:
        tokens = line.split()

        # Convert tokens to integers where applicable, ignoring non-integer tokens
        int_tokens = [int(token) for token in tokens if token.lstrip('-').isdigit()]

        # Update num_vars to the largest absolute integer value found
        if int_tokens:
            num_vars = max(num_vars, max(abs(token) for token in int_tokens))

        # Count the number of '0's in the line to estimate num_clauses
        zero_count = tokens.count('0')
        num_clauses = max(num_clauses, zero_count)

        # Update context_len to the maximum number of tokens in a line
        context_len = max(context_len, len(tokens))

    # Final adjustments
    num_clauses += 5  # Add 5 to the largest number of '0's
    context_len += 100  # Add 100 to the maximum number of tokens in a line

    return num_vars, num_clauses, context_len