from sops import *
import sops
import numpy as np
from typing import List, Dict, Tuple


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
                   + ['0', '[SEP]', '[UP]', '[BT]', '[BOS]', 'D', 'SAT', 'UNSAT'])
    idx: Dict[str, int] = {token: idx for idx, token in enumerate(vocab)}
    sop_logs: Dict[str, SOp] = {}
    sops.config["mean_exactness"] = mean_exactness
    # Initialize Base SOps
    tok_emb = OneHotTokEmb(idx).named("tok_emb")

    nearest_sep = nearest_token(tok_emb=tok_emb,
                                vocab=vocab,
                                targets=['0', '[SEP]', '[UP]', '[BT]'],
                                v=[indices, 'target'], mean_exactness=mean_exactness, max_id=context_len).named(
        "nearest_sep")

    p_i_sep_p, b_0, b_SEP, b_UP, b_BackTrack = (
        nearest_sep[:, 0].named("p_i_sep_p"),
        nearest_sep[:, 1].named("b_0"),
        nearest_sep[:, 2].named("b_SEP"),
        nearest_sep[:, 3].named("b_UP"),
        nearest_sep[:, 4].named("b_BackTrack"))

    p_i_D = nearest_token(tok_emb=tok_emb, vocab=vocab, targets=['D'], v=indices, mean_exactness=mean_exactness,
                          max_id=context_len).named("p_i_D")
    prev_pos = Id([p_i_sep_p, tok_emb[:, idx['D']]])[indices - 1].named("prev_pos")
    p_i_sep, b_decision = prev_pos[:, 0].named("p_i_sep"), prev_pos[:, 1].named("b_decision")

    d_i_sep = (indices - p_i_sep_p).named("d_i_sep")

    p_i_sep_2 = (p_i_sep * p_i_sep).named("p_i_sep_2")
    e_vars = tok_emb[:, : 2 * num_vars].named("e_vars")
    r_i_pre = Mean(q_sops=[p_i_sep_2, p_i_sep, ones],
                   k_sops=[-ones, 2 * p_i_sep, -p_i_sep_2],
                   v_sops=e_vars, exactness=mean_exactness).named("r_i_pre")
    r_i = (r_i_pre * (indices - p_i_sep)).named("r_i")

    p_i_sep_min = p_i_sep[p_i_sep_p].named("p_i_sep_min")

    p_i_min = (p_i_sep_min + d_i_sep + 4 * b_SEP).named("p_i_min")

    p_i_D_min = p_i_D[p_i_sep_p].named("p_i_D_min")

    b_exceed = (p_i_min > (p_i_D_min + 1)).named("b_exceed")

    b_D_min = (p_i_D_min == p_i_min + 1).named("b_D_min")

    sat_q = [r_i, ones]
    sat_k = [-r_i, (-nonsep_penalty) * (1 - tok_emb[:, idx['0']])]
    sat_v = is_bos
    b_sat = (Mean(sat_q, sat_k, sat_v, bos_weight=nonsep_penalty - 0.5, exactness=mean_exactness) > 0).named("b_sat")

    unsat_q = [t(r_i, num_vars, true_vec=(1, 0), false_vec=(0, 1), none_vec=(1, 1)), ones]
    unsat_k = sat_k
    unsat_v = 1 - is_bos
    b_cont = (Mean(unsat_q, unsat_k, unsat_v, bos_weight=nonsep_penalty - 0.5, exactness=mean_exactness) > 0).named(
        "b_cont")
    b_copy_p = (p_i_min < (p_i_sep_p - 1)).named("b_copy_p")

    orig_up_impl = True
    if orig_up_impl:
        up_q = [t(r_i, num_vars, true_vec=(0, 1), false_vec=(1, 0), none_vec=(0, 0)), ones]
        up_k = [r_i, (-nonsep_penalty) * (1 - tok_emb[:, idx['0']])]
        up_v = num_clauses * r_i
        o_up = Mean(up_q, up_k, up_v, bos_weight=nonsep_penalty + 1.5, exactness=mean_exactness).named("o_up")
    else:
        up_q = unsat_q
        up_k = unsat_k
        up_v = num_clauses * r_i
        o_up = Mean(up_q, up_k, up_v, bos_weight=nonsep_penalty - 1.5, exactness=mean_exactness).named("o_up")
    e_up = (GLUMLP(act_sops=(o_up - t(r_i, num_vars, true_vec=(1, 1), false_vec=(1, 1), none_vec=(0, 0))))
            - GLUMLP(act_sops=(o_up - 1))).named("e_up")

    heuristic_q = [t(r_i, num_vars, true_vec=(-10, 1), false_vec=(1, -10), none_vec=(0, 0)), ones]
    heuristic_k = up_k
    heuristic_v = r_i
    heuristic_o = SelfAttention(heuristic_q, heuristic_k, heuristic_v).named("heuristic_o")

    b_final = (b_exceed & b_decision).named("b_final")
    b_no_decision = (p_i_D <= p_i_sep).named("b_no_decision")
    b_BT_finish = ((p_i_D_min <= p_i_min) & b_BackTrack).named("b_BT_finish")
    e_BT = t(e_vars[p_i_D_min + 1], num_vars=num_vars, true_vec=(0, 1), false_vec=(1, 0), none_vec=(0, 0)).named("e_BT")
    p_i_min_index = (p_i_min + 1).named("p_i_min_index")
    e_copy = tok_emb[p_i_min_index].named("e_copy")
    b_unsat = (b_no_decision & b_cont).named("b_unsat")
    b_backtrack = (b_D_min & b_BackTrack).named("b_backtrack")
    b_copy = (b_copy_p & (1 - b_BT_finish)).named("b_copy")
    b_BT_token = (b_cont & (1 - tok_emb[:, idx['[BT]']])).named("b_BT_token")
    b_not_D = (1 - tok_emb[:, idx['D']]).named("b_not_D")
    e_unassigned = t(r_i, num_vars, true_vec=(0, 0), false_vec=(0, 0), none_vec=(1, 1)).named("e_unassigned")

    out = CPOutput(len(vocab), [(b_sat, idx['SAT'], 16),
                                (b_unsat, idx['UNSAT'], 15),
                                (b_BT_token, idx['[BT]'], 14),
                                #(b_final, idx['[UP]'], 13),
                                (b_backtrack, Pad(e_BT, len(vocab), idx['1']), 12),
                                (b_copy, e_copy, 6),
                                (None, Pad(e_up, len(vocab), idx['1']), 4),
                                (b_not_D, idx['D'], 3),
                                (None, Pad(e_unassigned + heuristic_o,
                                           out_dim=len(vocab), start_dim=idx['1']), 1)
                                ]).named("out")

    sop_logs = {
        "tok_emb": tok_emb,
        "nearest_sep": nearest_sep,
        "p_i_sep_p": p_i_sep_p,
        "b_0": b_0,
        "b_SEP": b_SEP,
        "b_UP": b_UP,
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
        "b_exceed": b_exceed,
        "b_D_min": b_D_min,
        "b_sat": b_sat,
        "b_cont": b_cont,
        "b_copy_p": b_copy_p,
        "o_up": o_up,
        "e_up": e_up,
        "b_final": b_final,
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
