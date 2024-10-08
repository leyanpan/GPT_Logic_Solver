from sops import *
import sops
import numpy as np
from typing import List, Dict, Tuple

def nearest_token_id(tok_emb: OneHotTokEmb, vocab: List[str], targets: List[str], indices: Indices=indices):
    # Get the token ids of the target tokens
    target_tok_ids = [vocab.index(target) for target in targets]

    # Get whether the current token is one of the target tokens by summing the one-hot embedding
    target_token_embs = Concat([tok_emb[:, target_tok_id] for target_tok_id in target_tok_ids])
    in_targets = target_token_embs.sum(axis=1)

    # Filter the indices to only include the target tokens
    filtered_index = indices * in_targets

    return filtered_index.max()


def nearest_token_new(tok_emb: OneHotTokEmb, vocab: List[str], targets: List[str], v: SOp | List[SOp],
                      mean_exactness=10, max_id=100, indices: PosEncSOp = indices):
    if not isinstance(v, list):
        v = [v]

    target_tok_ids = [vocab.index(target) for target in targets]
    target_tokens = Concat([tok_emb[:, target_tok_id] for target_tok_id in target_tok_ids])
    in_targets = Linear(target_tokens, np.ones((1, len(targets))))
    filtered_index = indices * in_targets

    new_v = []
    for v_i in v:
        if isinstance(v_i, SOp):
            new_v.append(v_i)
        elif v_i == 'target' or v_i == 'targets':
            new_v.append(target_tokens)
        else:
            raise ValueError('Unsupported value type')

    return Mean(ones, filtered_index, new_v, bos_weight=1, exactness=mean_exactness)


def nearest_token_orig(tok_emb: OneHotTokEmb, vocab: List[str], targets: List[str], v: SOp | List[SOp],
                       mean_exactness=10, max_id=100, indices: PosEncSOp = indices):
    """
    Given a sequence of tokens, find for each position the index of the nearest token in a target vocabulary and
    returns the value vector of that position.

    Parameters:
        tok_emb: The one-hot token embedding of the input tokens.
        vocab: the full set of tokens in the vocabulary, in order of their id in the token embedding matrix.
        targets: the target tokens that should be considered as a nearest token.
        v: The target vector to return at the index of the nearest token.

    Returns:
        nearest_v: the v vector at the index of the nearest token in the target set of vocabulary.
    """
    if not isinstance(v, list):
        v = [v]
    nearest_token_q = Const(1, dim=len(targets))
    target_indices = [vocab.index(target) for target in targets]
    nearest_token_k = Concat([tok_emb[:, target_index] for target_index in target_indices])
    new_v = []
    for v_i in v:
        if isinstance(v_i, SOp):
            new_v.append(v_i)
        elif v_i == 'target' or v_i == 'targets':
            new_v.append(nearest_token_k)
        else:
            raise ValueError('Unsupported value type')
    return Copy(nearest_token_q, nearest_token_k, Concat(new_v), bos_weight=1, exactness=mean_exactness, max_id=max_id, indices=indices)


use_new = True
nearest_token = nearest_token_new if use_new else nearest_token_orig


def t(encodings: SOp, num_vars, true_vec=(1, 0), false_vec=(0, 1), none_vec=(0, 0), ones: Ones = ones):
    mat = np.zeros((2 * num_vars, 2 * num_vars))
    true_vec_off = (true_vec[0] - none_vec[0], true_vec[1] - none_vec[1])
    false_vec_off = (false_vec[0] - none_vec[0], false_vec[1] - none_vec[1])
    for i in range(num_vars):
        true_id = i
        false_id = num_vars + i
        mat[true_id, true_id] = true_vec_off[0]
        mat[true_id, false_id] = false_vec_off[0]
        mat[false_id, true_id] = true_vec_off[1]
        mat[false_id, false_id] = false_vec_off[1]

    bias = np.zeros(2 * num_vars)
    bias[:num_vars] += none_vec[0]
    bias[num_vars:] = none_vec[1]

    return Linear([encodings, ones], np.hstack([mat.T, bias.reshape((-1, 1))]))


def dpll(num_vars, num_clauses, context_len, mean_exactness=10, nonsep_penalty=10, return_logs=False) -> Tuple[
    SOp, List, Dict[str, SOp]]:
    vocab: List = ([str(i) for i in range(1, num_vars + 1)]
                   + [str(-i) for i in range(1, num_vars + 1)]
                   + ['0', '[SEP]', '[BT]', '[BOS]', 'D', 'SAT', 'UNSAT'])
    idx: Dict[str, int] = {token: idx for idx, token in enumerate(vocab)}
    sop_logs: Dict[str, SOp] = {}
    sops.config["mean_exactness"] = mean_exactness
    # Initialize Base SOps
    tok_emb = OneHotTokEmb(idx).named("tok_emb")

    nearest_sep = nearest_token(tok_emb=tok_emb,
                                vocab=vocab,
                                targets=['0', '[SEP]', '[BT]'],
                                v=[indices, 'target'],
                                max_id=context_len).named(
        "nearest_sep")

    # The nearest (including self) separator token and whether
    # the previous separator token is '0', '[SEP]', '[UP]', '[BT]'
    p_i_sep_p, b_0, b_SEP, b_BackTrack = (
        nearest_sep[:, 0].named("p_i_sep_p"),
        nearest_sep[:, 1].named("b_0"),
        nearest_sep[:, 2].named("b_SEP"),
        nearest_sep[:, 3].named("b_BackTrack"))

    # The nearest 'D' token, which denotes the next token is a decision literal
    p_i_D = nearest_token(tok_emb=tok_emb, vocab=vocab, targets=['D'],
                          v=indices, max_id=context_len).named("p_i_D")

    prev_pos = Id([p_i_sep_p, tok_emb[:, idx['D']]])[indices - 1].named("prev_pos")
    # p_i_sep: The previous (excluding self) separator token
    p_i_sep = (prev_pos[:, 0] - is_bos).named("p_i_sep")

    # b_decision: whether the current position is a decision literal
    b_decision = prev_pos[:, 1].named("b_decision")

    # The distance to the nearest separator, i.e., the length of the current state
    d_i_sep = (indices - p_i_sep_p).named("d_i_sep")

    # Attention operation for representing the current clause/assignment as a bitvector of dimension 2d
    p_i_sep_2 = (p_i_sep * p_i_sep).named("p_i_sep_2")
    e_vars = tok_emb[:, : 2 * num_vars].named("e_vars")
    r_i_pre = Mean(q_sops=[p_i_sep_2, p_i_sep, ones],
                   k_sops=[-ones, 2 * p_i_sep, -p_i_sep_2],
                   v_sops=e_vars).named("r_i_pre")
    r_i = (r_i_pre * (indices - p_i_sep)).named("r_i")

    # The position of the previous (excluding self) separator token
    p_i_sep_min = p_i_sep[p_i_sep_p].named("p_i_sep_min")

    # The same position in the previous state. This is used for copying from the previous state
    p_i_min = (p_i_sep_min + d_i_sep + num_vars * b_SEP).named("p_i_min")

    # The position of the last decision in the previous state
    p_i_D_min = p_i_D[p_i_sep_p].named("p_i_D_min")

    # Is the next token the literal resulting from backtracking?
    b_D_min = (p_i_D_min == p_i_min + 1).named("b_D_min")

    # Check if the current assignment satisfies the formula (See Theorem Proof for justification)
    sat_q = [r_i, ones]
    sat_k = [-r_i, (-nonsep_penalty) * (1 - tok_emb[:, idx['0']])]
    sat_v = is_bos
    b_sat = (Mean(sat_q, sat_k, sat_v, bos_weight=nonsep_penalty - 0.5, exactness=mean_exactness) > 0).named("b_sat")

    # Check if the current assignment contracdicts the formula (See Theorem Proof for justification)
    unsat_q = [t(r_i, num_vars, true_vec=(1, 0), false_vec=(0, 1), none_vec=(1, 1)), ones]
    unsat_k = sat_k
    unsat_v = 1 - is_bos
    b_cont = (Mean(unsat_q, unsat_k, unsat_v, bos_weight=nonsep_penalty - 0.5, exactness=mean_exactness) > 0).named(
        "b_cont")
    b_copy_p = (p_i_min < (p_i_sep_p - 1)).named("b_copy_p")

    # Assume orig_up_impl is defined somewhere above
    orig_up_impl = False  # or False, depending on the desired implementation

    # Compute both versions regardless of orig_up_impl
    # Original implementation
    up_q_orig = [t(r_i, num_vars, true_vec=(0, 1), false_vec=(1, 0), none_vec=(0, 0)), ones]
    up_k_orig = [r_i, (-nonsep_penalty) * (1 - tok_emb[:, idx['0']])]
    up_v_orig = num_clauses * r_i
    o_up_orig = Mean(up_q_orig, up_k_orig, up_v_orig, bos_weight=nonsep_penalty + 1.5, exactness=mean_exactness).named(
        "o_up_orig")

    # New implementation
    up_q_new = unsat_q
    up_k_new = unsat_k
    up_v_new = num_clauses * r_i
    o_up_new = Mean(up_q_new, up_k_new, up_v_new, bos_weight=nonsep_penalty - 1.5, exactness=mean_exactness).named(
        "o_up_new")

    # Compute e_up for both implementations
    e_up_orig = (
            GLUMLP(act_sops=(o_up_orig - t(r_i, num_vars, true_vec=(1, 1), false_vec=(1, 1), none_vec=(0, 0))))
            - GLUMLP(act_sops=(o_up_orig - 1))
    ).named("e_up_orig")

    e_up_new = (
            GLUMLP(act_sops=(o_up_new - t(r_i, num_vars, true_vec=(1, 1), false_vec=(1, 1), none_vec=(0, 0))))
            - GLUMLP(act_sops=(o_up_new - 1))
    ).named("e_up_new")

    # Select e_up and o_up based on orig_up_impl
    if orig_up_impl:
        e_up = e_up_orig
        o_up = o_up_orig
    else:
        e_up = e_up_new
        o_up = o_up_new

    # Heuristic for decision literal selection: Find the most common literal in remaining clauses
    heuristic_q = [t(r_i, num_vars, true_vec=(-10, 1), false_vec=(1, -10), none_vec=(0, 0)), ones]
    heuristic_k = [r_i, (-nonsep_penalty) * (1 - tok_emb[:, idx['0']])]
    heuristic_v = r_i
    heuristic_o = SelfAttention(heuristic_q, heuristic_k, heuristic_v).named("heuristic_o")

    # Whether the current assignment contains no decision literal
    b_no_decision = (p_i_D <= p_i_sep).named("b_no_decision")

    # Whether Backtracking is finished
    b_BT_finish = ((p_i_D_min <= p_i_min) & b_BackTrack).named("b_BT_finish")

    # The negation of the last decision literal in the previous state
    e_BT = t(e_vars[p_i_D_min + 1], num_vars=num_vars, true_vec=(0, 1), false_vec=(1, 0), none_vec=(0, 0)).named("e_BT")

    # The next index in the previous state for copying
    p_i_min_index = (p_i_min + 1).named("p_i_min_index")

    # The next token in the previous state for copying
    e_copy = tok_emb[p_i_min_index].named("e_copy")

    # Whether we've decided that the formula is UNSAT
    b_unsat = (b_no_decision & b_cont).named("b_unsat")

    # Whether we're negativing the last decision literal for backtracking
    b_backtrack = (b_D_min & b_BackTrack).named("b_backtrack")

    # Whether we're copying tokens from the previous state
    b_copy = (b_copy_p & (1 - b_BT_finish)).named("b_copy")

    b_BT_token = (b_cont & (1 - tok_emb[:, idx['[BT]']])).named("b_BT_token")
    b_not_D = (1 - tok_emb[:, idx['D']]).named("b_not_D")
    e_unassigned = t(r_i, num_vars, true_vec=(0, 0), false_vec=(0, 0), none_vec=(1, 1)).named("e_unassigned")

    out = CPOutput(len(vocab), [(b_sat, idx['SAT'], 16),
                                (b_unsat, idx['UNSAT'], 15),
                                (b_BT_token, idx['[BT]'], 14),
                                (b_backtrack, Pad(e_BT, len(vocab), idx['1']), 12),
                                (b_copy, e_copy, 6),
                                (None, Pad(e_up, len(vocab), idx['1']), 4),
                                (b_not_D, idx['D'], 3),
                                (None, Pad(e_unassigned + heuristic_o,
                                           out_dim=len(vocab), start_dim=idx['1']), 1)
                                ]).named("out")

    # Update sop_logs to include variables from both paths
    sop_logs = {
        "tok_emb": tok_emb,
        "nearest_sep": nearest_sep,
        "p_i_sep_p": p_i_sep_p,
        "b_0": b_0,
        "b_SEP": b_SEP,
        "b_BackTrack": b_BackTrack,
        "p_i_D": p_i_D,
        "prev_pos": prev_pos,
        "p_i_sep": p_i_sep,
        "b_decision": b_decision,
        "d_i_sep": d_i_sep,
        "p_i_sep_2": p_i_sep_2,
        "e_vars": e_vars,
        "r_i_pre": r_i_pre,
        "r_i": r_i,
        "p_i_sep_min": p_i_sep_min,
        "p_i_min": p_i_min,
        "p_i_D_min": p_i_D_min,
        "b_D_min": b_D_min,
        "b_sat": b_sat,
        "unsat_q": Linear(unsat_q),
        "unsat_k": Linear(unsat_k),
        "b_cont": b_cont,
        "b_copy_p": b_copy_p,
        # Original implementation variables
        "up_q_orig": up_q_orig,
        "up_k_orig": up_k_orig,
        "up_v_orig": up_v_orig,
        "o_up_orig": o_up_orig,
        "e_up_orig": e_up_orig,
        # New implementation variables
        "up_q_new": up_q_new,
        "up_k_new": up_k_new,
        "up_v_new": up_v_new,
        "o_up_new": o_up_new,
        "e_up_new": e_up_new,
        # Selected outputs
        "o_up": o_up,
        "e_up": e_up,
        "b_no_decision": b_no_decision,
        "b_BT_finish": b_BT_finish,
        "e_BT": e_BT,
        "p_i_min_index": p_i_min_index,
        "e_copy": e_copy,
        "b_unsat": b_unsat,
        "b_backtrack": b_backtrack,
        "b_copy": b_copy,
        "b_BT_token": b_BT_token,
        "b_not_D": b_not_D,
        "e_unassigned": e_unassigned,
        "out": out
    }

    return out, vocab, sop_logs
