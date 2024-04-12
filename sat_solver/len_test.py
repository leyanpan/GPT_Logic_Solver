import sys
import numpy
import matplotlib.pyplot as plt

fn = sys.argv[1]
max_tok = 0
toks = []
with open(fn) as f:
    for line in f:
        # tok = len(line.strip().replace("-", "- ").split())
        # no space after -:
        tok = len(line.strip().split())
        toks.append(tok)
        if tok > max_tok:
            max_tok = tok

print(f"Max token length: {max_tok}")

# Plot distribution of token lengths
plt.hist(toks, bins=20)
plt.xlabel('Token Length')
plt.ylabel('Frequency')
plt.title('Distribution of Token Lengths')
plt.show()