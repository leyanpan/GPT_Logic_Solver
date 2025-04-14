# evaluate_all.py

from .dpll import dpll
from .compiler import compile_model
import torch
import argparse
import os
import numpy as np
import re
import tqdm

from .eval_utils import evaluate_model, infer_args_from_file  # Import functions

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the DPLL model on all files in a directory")

    parser.add_argument('directory', type=str, help="Directory containing dataset files")

    # Arguments for the dpll function
    parser.add_argument('-v', '--num_vars', type=int, default=None, help="Number of variables")
    parser.add_argument('-c', '--num_clauses', type=int, default=None, help="Number of clauses")
    parser.add_argument('-l', '--context_len', type=int, default=None, help="Context length")

    # Modify the argument parser to accept a list of mean_exactness values
    parser.add_argument('-m', '--mean_exactness', type=float, nargs='+', default=[50],
                        help="List of Mean exactness values (default: [50])")

    parser.add_argument('-p', '--nonsep_penalty', type=float, default=50, help="Non-separation penalty (default: 50)")
    parser.add_argument('-n', '--samples', type=int, default=None, help="Number of samples (default: all samples)")
    parser.add_argument('--max_samples', type=int, default=None, help="Maximum number of samples to evaluate per file")
    parser.add_argument('-s', '--state', action='store_true', help="State-based generation method")
    parser.add_argument('-a', '--abstract', action='store_true', help="Abstract generation method")

    return parser.parse_args()

def remove_states_followed_by_up(s: str) -> str:
    """
    Remove all states followed by the [UP] token and the [UP] token themselves.
    """
    separators = r'\[SEP\]|\[BT\]|\[UP\]'
    parts = re.split(f'({separators})', s)
    output = []
    i = 0

    while i < len(parts):
        part = parts[i]
        if part == '[UP]':
            if output and re.match(separators, output[-1]) is None:
                output.pop()
        elif part.strip() != '[UP]':
            output.append(part)
        i += 1

    return ''.join(output).strip()

if __name__ == "__main__":
    args = parse_arguments()
    # Iterate over all files in the directory
    for filename in os.listdir(args.directory):
        file_path = os.path.join(args.directory, filename)

        if os.path.isfile(file_path) and filename.endswith('.txt'):
            print(f"Evaluating file: {filename}")

            # Read the file content
            with open(file_path, 'r') as f:
                samples = f.readlines()

            # Limit the number of samples if --max_samples is specified
            if args.max_samples is not None:
                samples = samples[:args.max_samples]

            # Infer arguments from the file if they are not provided
            if args.num_vars is None or args.num_clauses is None or args.context_len is None:
                inferred_num_vars, inferred_num_clauses, inferred_context_len = infer_args_from_file(samples)
                num_vars = args.num_vars if args.num_vars is not None else inferred_num_vars
                num_clauses = args.num_clauses if args.num_clauses is not None else inferred_num_clauses
                context_len = args.context_len if args.context_len is not None else inferred_context_len
            else:
                num_vars = args.num_vars
                num_clauses = args.num_clauses
                context_len = args.context_len

            # Print the inferred or provided values
            print(f"Number of Variables: {num_vars}")
            print(f"Number of Clauses: {num_clauses}")
            print(f"Context Length: {context_len}")

            # Call the dpll function with the parsed arguments
            dpll_sop, id_to_tokens, sop_logs = dpll(
                num_vars=num_vars,
                num_clauses=num_clauses,
                context_len=context_len,
                mean_exactness=10,
                nonsep_penalty=args.nonsep_penalty,
                return_logs=True,
                heuristic=False
            )

            num_sample_correct = 0
            total_samples = 0

            num_step_correct = 0
            total_steps = 0

            token_to_id = {token: i for i, token in enumerate(id_to_tokens)}
            for sample in tqdm.tqdm(samples):
                sample = remove_states_followed_by_up(sample)
                tokens = sample.split()
                tokens = [token if token in id_to_tokens else '[SEP]' for token in tokens]
                if tokens[0] != '[BOS]':
                    tokens = ['[BOS]'] + tokens
                sep_pos = tokens.index('[SEP]')
                outputs = dpll_sop.abstract_eval(tokens)
                trace_pred = tokens[sep_pos + 1:]
                trace_scores = outputs[sep_pos:-1]
                token_ids = [token_to_id.get(token, token_to_id['[SEP]']) for token in trace_pred]
                pred_token_scores = trace_scores[np.arange(len(trace_pred)), token_ids]
                max_scores = np.max(trace_scores, axis=1)
                full_correct = np.allclose(pred_token_scores, max_scores, atol=0.1)
                num_correct = np.isclose(pred_token_scores, max_scores, atol=0.1)

                if full_correct:
                    num_sample_correct += 1
                total_samples += 1
                num_step_correct += np.sum(num_correct)
                total_steps += len(num_correct)

            # Print statistics for file
            print(f"File: {filename}")
            print(f"Sample Accuracy: {num_sample_correct / total_samples}")
            print(f"Step Accuracy: {num_step_correct / total_steps}")
            print(f"Number of Samples Evaluated: {total_samples}")

        else:
            print(f"Skipping {filename}")
