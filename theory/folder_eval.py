# evaluate_all.py

from dpll import dpll
from compiler import compile_model
import torch
import argparse
import os

from eval_utils import evaluate_model, infer_args_from_file  # Import functions

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
    parser.add_argument('-s', '--state', action='store_true', help="State-based generation method")
    parser.add_argument('-a', '--abstract', action='store_true', help="Abstract generation method")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # Iterate over all mean_exactness values
    for mean_exactness_value in args.mean_exactness:
        print(f"Evaluating with mean_exactness = {mean_exactness_value}")

        # Iterate over all files in the directory
        for filename in os.listdir(args.directory):
            file_path = os.path.join(args.directory, filename)
            if os.path.isfile(file_path) and filename.endswith('.txt'):
                print(f"Evaluating file: {filename}")

                # Read the file content
                with open(file_path, 'r') as f:
                    samples = f.readlines()

                if args.samples is not None:
                    samples = samples[:args.samples]

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
                dpll_sop, tokens, sop_logs = dpll(
                    num_vars=num_vars,
                    num_clauses=num_clauses,
                    context_len=context_len,
                    mean_exactness=mean_exactness_value,
                    nonsep_penalty=args.nonsep_penalty,
                    return_logs=True
                )

                dpll_model = compile_model(dpll_sop, tokens, context_len)

                if torch.cuda.is_available():
                    dpll_model.cuda()

                # Evaluate the model
                accuracy, max_clauses, max_cot_len, max_bt, avg_cot_len, avg_bt = evaluate_model(
                    dpll_model, samples, tokens, dpll_sop, args
                )

                # Prepare the result string
                result_str = (
                    f"Evaluating file: {filename}\n"
                    f"Mean Exactness: {mean_exactness_value}\n"
                    f"Number of Variables: {num_vars}\n"
                    f"Number of Clauses: {num_clauses}\n"
                    f"Context Length: {context_len}\n"
                    f"Accuracy: {accuracy * 100:.2f}%\n"
                    f"Maximum Number of Clauses in Prompt: {max_clauses}\n"
                    f"Maximum Chain-of-Thought Length: {max_cot_len}\n"
                    f"Maximum Number of Backtracking Steps: {max_bt}\n"
                    f"Average Chain-of-Thought Length: {avg_cot_len:.2f}\n"
                    f"Average Number of Backtracking Steps: {avg_bt:.2f}\n"
                )

                print(result_str)

                # Write the results to a file named eval_out_{filename}_{mean_exactness}.txt
                output_filename = f"eval_out_{filename.replace('.txt', '')}_{mean_exactness_value}.txt"
                with open(output_filename, 'w') as outfile:
                    outfile.write(result_str)
            else:
                print(f"Skipping {filename}")