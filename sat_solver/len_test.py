import sys

fn = sys.argv[1]
max_tok = 0
with open(fn) as f:
    for line in f:
        # tok = len(line.strip().replace("-", "- ").split())
        # no space after -:
        tok = len(line.strip().split())
        if tok > max_tok:
            max_tok = tok

print(max_tok)