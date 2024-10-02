# evaluation_script.py

from dpll import dpll
from compiler import compile_model
import torch
import argparse
from typing import List
import os

from eval_utils import evaluate_model, infer_args_from_file  # Import the evaluate_model function

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
    parser.add_argument('-a', '--abstract', action='store_true', help="Abstract generation method")

    return parser.parse_args()


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

    # Use the evaluate_model function from eval_utils
    accuracy, max_clauses, max_cot_len, max_bt, avg_cot_len, avg_bt = evaluate_model(
        dpll_model, samples, tokens, dpll_sop, args, print_results=True
    )

    # Print the results
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Maximum Number of Clauses in Prompt: {max_clauses}")
    print(f"Maximum Chain-of-Thought Length: {max_cot_len}")
    print(f"Maximum Number of Backtracking Steps: {max_bt}")
    print(f"Average Chain-of-Thought Length: {avg_cot_len:.2f}")
    print(f"Average Number of Backtracking Steps: {avg_bt:.2f}")