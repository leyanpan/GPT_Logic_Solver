from dpll import dpll
from compiler import compile_model
import torch

import argparse
from typing import Tuple, List, Dict

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse arguments for the DPLL function")

    parser.add_argument('filename', type=str, help="Dataset File")

    # Arguments for the dpll function
    parser.add_argument('-v', '--num_vars', type=int, default=None, help="Number of variables")
    parser.add_argument('-c', '--num_clauses', type=int, default=None, help="Number of clauses")
    parser.add_argument('-l', '--context_len', type=int, default=None, help="Context length")

    # Optional arguments with default values in dpll
    parser.add_argument('-m', '--mean_exactness', type=float, default=50, help="Mean exactness (default: 50)")
    parser.add_argument('-p', '--nonsep_penalty', type=float, default=50, help="Non-separation penalty (default: 50)")
    parser.add_argument('-n', '--samples', type=int, default=None, help="Number of samples (default: all samples)")
    parser.add_argument('-s', '--state', action='store_true', help="State-based generation method")

    return parser.parse_args()

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

if __name__ == "__main__":
    args = parse_arguments()

    # Read the file content
    with open(args.filename, 'r') as f:
        samples = f.readlines()

    if args.samples is not None:
        samples = samples[:args.samples]

    # Infer arguments from the file if they are not provided
    if args.num_vars is None or args.num_clauses is None or args.context_len is None:
        inferred_num_vars, inferred_num_clauses, inferred_context_len = infer_args_from_file(samples)
        if args.num_vars is None:
            args.num_vars = inferred_num_vars
        if args.num_clauses is None:
            args.num_clauses = inferred_num_clauses
        if args.context_len is None:
            args.context_len = inferred_context_len

    # Print the inferred or provided values
    print(f"Number of Variables: {args.num_vars}")
    print(f"Number of Clauses: {args.num_clauses}")
    print(f"Context Length: {args.context_len}")

    # Call the dpll function with the parsed arguments
    dpll_sop, tokens, sop_logs = dpll(
        num_vars=args.num_vars,
        num_clauses=args.num_clauses,
        context_len=args.context_len,
        mean_exactness=args.mean_exactness,
        nonsep_penalty=args.nonsep_penalty,
        return_logs=True
    )

    dpll_model = compile_model(dpll_sop, tokens, args.context_len)

    if torch.cuda.is_available():
        dpll_model.cuda()

    correct_predictions = 0
    total_samples = len(samples)

    max_clauses = 0  # Max number of clauses (number of '0' tokens in the prompt)
    max_cot_len = 0  # Max length of Chain-of-Thought (len difference between prompt and generated tokens)
    max_bt = 0       # Maximum number of backtracking (number of generated '[BT]' tokens)

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

        # Generate completion using dpll_model
        if args.state:
            completion_tokens = dpll_model.state_generate(prompt_tokens)
        else:
            completion_tokens = dpll_model.generate(prompt_tokens)
        print(" ".join(completion_tokens))

        # Update max_cot_len (len difference between prompt and completion)
        cot_len = len(completion_tokens) - len(prompt_tokens)
        if cot_len > max_cot_len:
            max_cot_len = cot_len

        # Update max_bt (number of '[BT]' tokens in completion)
        num_bt_tokens = completion_tokens.count('[BT]')
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
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Print the requested statistics
    print(f"Maximum Number of Clauses in Prompt: {max_clauses}")
    print(f"Maximum Chain-of-Thought Length: {max_cot_len}")
    print(f"Maximum Number of Backtracking Steps: {max_bt}")