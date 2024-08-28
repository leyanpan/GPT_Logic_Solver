import os
import sys

def max_token_count(fn):
    # Count the maximum number of tokens for a line in a file
    max_tokens = 0
    with open(fn, 'r') as f:
        for line in f:
            tokens = len(line.split())
            if tokens > max_tokens:
                max_tokens = tokens

    return max_tokens

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: max_token_count.py FILE")

    fn = sys.argv[1]
    if not os.path.exists(fn):
        sys.exit(f"Error: {fn} not found")

    max_tokens = max_token_count(fn)
    print(f"Max tokens: {max_tokens}")

if __name__ == "__main__":
    main()