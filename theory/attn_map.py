from .dpll import dpll
from typing import List, Dict
from .sops import SelfAttention
import numpy as np
import matplotlib.pyplot as plt


NUM_VARS = 20
NUM_CLAUSES = 100
CONTEXT_LEN = 800

dpll_sop, vocab, sops = dpll(NUM_VARS, NUM_CLAUSES, CONTEXT_LEN)

sat_sop, unsat_sop, up_sop_new, up_sop_orig = sops["b_sat"], sops["b_cont"], sops["o_up_orig"], sops["o_up_new"]

def get_logic_attn_maps(input_tokens: List[str]):
    sops_names = ['raw_sat', 'raw_cont', 'o_up_orig', 'o_up_new']
    attn_maps = {}
    for sop_name in sops_names:
        sop : SelfAttention = sops[sop_name]
        attn_map = sop.get_attn_weight(input_tokens)
        processed_attn = post_process_attn_map(attn_map, input_tokens, bos_token=True)
        attn_maps[sop_name] = processed_attn

    return attn_maps

def post_process_attn_map(attn_map: np.ndarray, input_tokens: List[str], bos_token=True) -> np.ndarray:
    """
    Post processes an attention map by summing over slices of the last dimension.
    The operation is applied to the last two dims of `attn_map` (assumed to be (rows, columns))
    so that extra leading dimensions are preserved.

    Parameters:
        attn_map (np.ndarray): An array with shape (..., rows, cols). The operation
                                 will be performed on each matrix of shape (rows, cols)
                                 in the extra dimensions.
        input_tokens (List[str]): A list of tokens whose length should match the number of
                                  columns (cols) of attn_map.
        bos_token (bool): Whether to skip the first column (e.g. beginning-of-sequence token).

    Returns:
        np.ndarray: Processed attention map of shape (..., rows, num_clauses+1) where each
                    clause is obtained by a cumulative sum over columns and the last column
                    provides the "remaining" attention (1 minus the sum over clauses).
    """

    sep_idx = input_tokens.index('[SEP]')
    attn_map = attn_map[..., sep_idx + 1:, :]

    # Unpack the shape so that rows and cols refer to the last two dims.
    *prefix, rows, cols = attn_map.shape

    # Basic check: input_tokens should correspond to the column count.
    if len(input_tokens) != cols:
        raise ValueError("Length of input_tokens must match the number of columns in attn_map.")


    # Determine the starting column (skip the beginning-of-sequence token if needed)
    start = 1 if bos_token else 0

    # Initialize a cumulative sum vector for each "matrix" with shape (..., rows, 1)
    cur_vec = np.zeros((*prefix, rows, 1), dtype=attn_map.dtype)
    ret_vecs = []

    # Loop over the columns (last dimension) from `start` to cols.
    for i in range(start, cols):
        # Extract the entire i-th column from the last two dims.
        # attn_map[..., :, i] has shape (*prefix, rows); we add an axis to become (*prefix, rows, 1)
        col_data = attn_map[..., :, i][..., None]
        cur_vec += col_data

        # When the condition is met, append the current accumulated vector and reset.
        if input_tokens[i] == '0':
            ret_vecs.append(cur_vec)
            cur_vec = np.zeros((*prefix, rows, 1), dtype=attn_map.dtype)

    # Convert list of arrays to a single array.
    # Each element in ret_vecs has shape (*prefix, rows, 1) and we concatenate along the last axis.
    clause_attns = np.concatenate(ret_vecs, axis=-1)  # Shape: (*prefix, rows, num_clauses)

    # Compute "remaining" attention per row: 1 minus the sum over the clause columns.
    clause_sum = clause_attns.sum(axis=-1, keepdims=True)  # Sum over clause columns (last axis)
    remaining_attn = 1 - clause_sum  # Shape: (*prefix, rows, 1)

    # Concatenate clause attentions with the remaining attention as the last "clause"
    processed_attn = np.concatenate([clause_attns, remaining_attn], axis=-1)

    return processed_attn


if __name__ == "__main__":
    TEST_INPUT = "[BOS] 6 2 1 0 -4 -3 6 0 6 1 5 0 6 4 2 0 -6 4 3 0 -2 -4 6 0 1 4 -2 0 3 -1 6 0 4 6 -1 0 -3 5 -6 0 -3 -4 -6 0 3 -2 -4 0 -1 4 -5 0 4 2 -3 0 -1 -2 -5 0 -5 -4 3 0 4 1 -2 0 -3 -2 -6 0 -6 1 5 0 2 -3 -1 0 6 5 -1 0 6 5 1 0 -1 4 -3 0 -6 4 -5 0 -6 2 1 0 [SEP] D 5 D 4 3 6 [BT] D 5 -4 -1 -2 6 [BT] -5 D -1 6 [BT] -5 1 6 -3 4 -2 SAT"
    TEST_INPUT = TEST_INPUT.split(" ")
    attn_maps = get_logic_attn_maps(TEST_INPUT)

    visualize_matrix = attn_maps["o_up_new"]
    # visualize the matrix
    plt.imshow(visualize_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("Attention Map for raw_sat")
    plt.xlabel("Token Position")
    plt.ylabel("Attention Weights")
    plt.tight_layout()
    plt.savefig("attention_map.png")  # Save as image
    plt.close()