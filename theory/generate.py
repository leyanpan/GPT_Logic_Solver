from dpll import dpll
from compiler import compile_model
import torch
import argparse

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPLL Model Interactive Prompt")

    # Arguments for the dpll function with default values
    parser.add_argument('-v', '--num_vars', type=int, default=10, help="Number of variables (default: 10)")
    parser.add_argument('-c', '--num_clauses', type=int, default=50, help="Number of clauses (default: 50)")
    parser.add_argument('-l', '--context_len', type=int, default=200, help="Context length (default: 200)")

    # Optional arguments with default values
    parser.add_argument('-m', '--mean_exactness', type=float, default=50, help="Mean exactness (default: 50)")
    parser.add_argument('-p', '--nonsep_penalty', type=float, default=50, help="Non-separation penalty (default: 50)")
    parser.add_argument('-s', '--state', action='store_true', help="Use state-based generation method")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # Build the DPLL model
    dpll_sop, tokens, sop_logs = dpll(
        num_vars=args.num_vars,
        num_clauses=args.num_clauses,
        context_len=args.context_len,
        mean_exactness=args.mean_exactness,
        nonsep_penalty=args.nonsep_penalty,
        return_logs=True
    )

    dpll_model = compile_model(dpll_sop, tokens, args.context_len)

    print(dpll_model.summary())

    # Move model to GPU if available
    if torch.cuda.is_available():
        dpll_model.cuda()

    print("DPLL Model is ready. Enter your prompts below (type 'exit' to quit).")

    while True:
        # Get user input
        user_input = input("Enter a prompt: ")

        # Exit condition
        if user_input.lower() == 'exit':
            print("Exiting the prompt loop. Goodbye!")
            break

        # Process the prompt
        prompt_tokens = user_input.strip().split()

        # Ensure the prompt starts with [BOS]
        if not prompt_tokens or prompt_tokens[0] != "[BOS]":
            prompt_tokens.insert(0, "[BOS]")

        # Ensure the prompt ends with [SEP]
        if "[SEP]" not in prompt_tokens:
            prompt_tokens.append("[SEP]")

        # Generate completion using the DPLL model
        if args.state:
            completion_tokens = dpll_model.state_generate(prompt_tokens)
        else:
            completion_tokens = dpll_model.generate(prompt_tokens)

        # Remove the original prompt from the generated sequence
        generated_tokens = completion_tokens[len(prompt_tokens):]

        # Output the generated sequence without the original prompt
        print("\nGenerated Output:")
        print(" ".join(generated_tokens))
        print("-" * 50)