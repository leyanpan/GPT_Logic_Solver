import numpy as np
from typing import List, Tuple, Dict
from copy import deepcopy

import torch

from sops import SOp, SelfAttention, GLUMLP, Linear, PosEncSOp, TokEmbSOp, Id
from model import TransformerModel, TransformerBlock


def topological_sort(start_sop: SOp) -> List[SOp]:
    # Initialize the graph structure
    child_sops = []  # This will store the topologically sorted SOps
    indegree = {}  # SOp -> number of incoming edges
    graph = {}  # SOp -> list of dependent SOps

    # Collect all SOps and initialize graph and indegree counts
    all_sops = set()
    queue = [start_sop]

    while queue:
        sop = queue.pop(0)
        if sop not in all_sops:
            all_sops.add(sop)
            queue.extend(sop.deps)

    for sop in all_sops:
        indegree[sop] = 0
        graph[sop] = []

    for sop in all_sops:
        for dep in sop.deps:
            graph[sop].append(dep)
            indegree[dep] += 1

    # Kahn's algorithm for topological sort
    queue = [sop for sop in all_sops if indegree[sop] == 0]

    while queue:
        sop = queue.pop(0)
        child_sops.append(sop)
        for dependent in graph[sop]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                queue.append(dependent)

    return child_sops


def flatten_linear(lin_sop: Linear) -> Linear:
    # Flatten Linear SOp such that it has no Linear dependencies. This Operation is in-place

    # Base Case 0: If the Linear SOp have no Linear dependencies, return itself
    if all(not isinstance(dep, Linear) for dep in lin_sop.deps):
        return lin_sop

    # Flatten Linear SOp dependencies
    new_deps: List[SOp] = []
    flattened_deps: List[SOp] = []

    for dep in lin_sop.deps:
        if isinstance(dep, Linear):
            flattened_dep = flatten_linear(dep)
            flattened_deps.append(flattened_dep)
            new_deps += flattened_dep.deps
        else:
            flattened_deps.append(dep)
            new_deps.append(dep)

    # All deps in new_deps should not be Linear
    assert all(not isinstance(dep, Linear) for dep in new_deps)
    new_in_dim = sum(dep.dim for dep in new_deps)

    # Start constucting the new weight matrix
    new_w = np.zeros((lin_sop.dim, new_in_dim))
    cur_dim_flat = 0
    cur_dim_orig = 0

    for orig_dep, flattened_dep in zip(lin_sop.deps, flattened_deps):
        # orig_dep: original dependency of lin_sop before flattening
        # flattened_dep: flattened dependency of lin_sop if they are linear, otherwise orig_dep
        if isinstance(flattened_dep, Linear):
            # We need to incorporate the weight matrix of the flattened dependency into the new weight matrix
            dep_in_dim = flattened_dep.w.shape[1]
            compound_weights = lin_sop.w[:, cur_dim_orig:cur_dim_orig + orig_dep.dim] @ flattened_dep.w
            new_w[:, cur_dim_flat:cur_dim_flat + dep_in_dim] += compound_weights
            cur_dim_flat += dep_in_dim
        else:
            assert orig_dep == flattened_dep
            orig_weights = lin_sop.w[:, cur_dim_orig:cur_dim_orig + orig_dep.dim]
            new_w[:, cur_dim_flat:cur_dim_flat + flattened_dep.dim] += orig_weights
            cur_dim_flat += flattened_dep.dim
        cur_dim_orig += orig_dep.dim

    return Linear(new_deps, new_w)


def flatten_all_linear(sop: SOp) -> SOp:
    if isinstance(sop, Linear):
        sop = flatten_linear(sop)
    for i, dep in enumerate(sop.deps):
        sop.deps[i] = flatten_all_linear(dep)
    return sop


def compile_model(out_sop: SOp, vocab: List[str], max_seq_len: int, min_mlp_dim=5, min_head_dim=5, return_alloc=False) -> TransformerModel | Tuple[TransformerModel, Dict, SOp]:
    # Since we need an unembedding matrix, the last SOp must be Linear
    if not isinstance(out_sop, Linear):
        out_sop = Id(out_sop)

    # This ensures that all Linear SOps are either direct deps of SelfAttention or MLP
    flat_sop: Linear = flatten_all_linear(deepcopy(out_sop))

    # Topological sort to find all child SOps
    child_sops = topological_sort(flat_sop)

    # Find all Token and Positional Embeddings
    tok_embs = [sop for sop in child_sops if isinstance(sop, TokEmbSOp)]
    pos_encs = [sop for sop in child_sops if isinstance(sop, PosEncSOp)]

    # There should be at least one Token and Positional Embedding
    if not tok_embs:
        print("Warning: No token embeddings found.")
    if not pos_encs:
        print("Warning: No positional encodings found.")

    # Assign residual stream allocations to each SOp that's not LinearTransform, in reverse order
    embed_dim = 0
    residual_alloc: Dict[SOp, Tuple[int, int]] = {}  # sop -> (start, end)
    for sop in reversed(child_sops):
        if not isinstance(sop, Linear):
            residual_alloc[sop] = (embed_dim, embed_dim + sop.dim)
            embed_dim += sop.dim

    # Topological sort of child_sops to assign each SelfAttention and GLUMLP to a layer
    layers: List[Tuple[List[SelfAttention], List[GLUMLP]]] = []

    # We now use the topologically sorted SOps to assign each SelfAttention and GLUMLP to the lowest possible layer
    sop_to_layer = {}

    # Assign pos and tok embeddings to layer -1
    for pos_enc in pos_encs:
        sop_to_layer[pos_enc] = -1

    for tok_emb in tok_embs:
        sop_to_layer[tok_emb] = -1

    # Process child_sops in reverse order
    for sop in reversed(child_sops):
        # Only consider SelfAttention and GLUMLP SOps
        if isinstance(sop, GLUMLP) or isinstance(sop, SelfAttention):
            # Determine the minimum layer for the current sop
            min_layer = 0
            # All deps should be Linear
            assert all(
                isinstance(dep, Linear) for dep in sop.deps), f"{sop.__class__} should only have Linear dependencies."

            for lin_dep in sop.deps:
                for dep in lin_dep.deps:
                    # All deps of Linear SOp should not be Linear after flattening
                    assert not isinstance(dep,
                                          Linear), f"Linear Flattening Failed: Linear SOp should not have Linear dependencies, but have dependency of type {dep.__class__}."
                    assert dep in residual_alloc, f"Dependency {dep} of {lin_dep} has not been assigned a residual stream allocation."
                    assert dep in sop_to_layer, f"Topological Sort Failed: Dependency {dep} of {lin_dep} has not been assigned to a layer."
                    assert isinstance(dep, (GLUMLP, SelfAttention, PosEncSOp,
                                            TokEmbSOp)), f"Dependency {dep} of {lin_dep} is not a valid layer type."

                    if isinstance(sop, GLUMLP) and isinstance(dep, SelfAttention):
                        # MLP can share the same layer as SelfAttention
                        min_layer = max(min_layer, sop_to_layer[dep])
                    else:
                        min_layer = max(min_layer, sop_to_layer[dep] + 1)

            # Ensure the layers list is long enough to accommodate min_layer
            while len(layers) <= min_layer:
                layers.append(([], []))

            # Assign sop to the appropriate layer
            if isinstance(sop, SelfAttention):
                layers[min_layer][0].append(sop)
            elif isinstance(sop, GLUMLP):
                layers[min_layer][1].append(sop)

            # Record the layer assignment
            sop_to_layer[sop] = min_layer

    # Compute required hyperparameters
    num_layers = len(layers)
    num_heads = max(list(len(layer[0]) for layer in layers) + [1])
    head_dim = max(list(sop.hidden_size for layer in layers for sop in layer[0]) + [min_head_dim])
    mlp_dim = max(list(sum(mlp.hidden_size for mlp in layer[1]) for layer in layers) + [min_mlp_dim])
    activations = [mlp_sop.activation for mlp_sop in child_sops if isinstance(mlp_sop, GLUMLP)]
    out_size = flat_sop.dim
    assert all(act == activations[0] for act in activations), "All GLUMLP should have the same activation function."
    activation = activations[0] if activations else "relu"

    model = TransformerModel(vocab_size=len(vocab),
                             max_seq_len=max_seq_len,
                             embed_dim=embed_dim,
                             num_heads=num_heads,
                             num_layers=num_layers,
                             mlp_dim=mlp_dim,
                             hidden_size=head_dim * num_heads,
                             out_size=out_size,
                             activation=activation,
                             vocab=vocab)
    with torch.no_grad():
        model.zero_weights()

        # Assign weights of SOps to the model
        for layer_id, layer_sops in enumerate(layers):
            self_attns, glumlps = layer_sops
            model_layer: TransformerBlock = model.layers[layer_id]
            for head, sop in enumerate(self_attns):
                q_proj, k_proj, v_proj, o_proj = sop.compile_weights(residual_alloc=residual_alloc, embed_dim=embed_dim,
                                                                     head_dim=head_dim, head_alloc=head,
                                                                     num_heads=num_heads)
                # Convert to torch tensors
                q_proj, k_proj, v_proj, o_proj = map(torch.tensor, (q_proj, k_proj, v_proj, o_proj))
                model_layer.attn.increment_weights(q_proj, k_proj, v_proj, o_proj)
            hidden_alloc = 0
            for mlp_sop in glumlps:
                mlp_w_in, mlp_b_in, mlp_w_out, mlp_b_out = mlp_sop.compile_weights(residual_alloc=residual_alloc,
                                                                                   embed_dim=embed_dim, mlp_dim=mlp_dim,
                                                                                   hidden_alloc=(hidden_alloc,
                                                                                                 hidden_alloc + mlp_sop.hidden_size))
                # Convert to torch tensors
                mlp_w_in, mlp_b_in, mlp_w_out, mlp_b_out = map(torch.tensor, (mlp_w_in, mlp_b_in, mlp_w_out, mlp_b_out))
                model_layer.mlp.increment_weights(mlp_w_in, mlp_b_in, mlp_w_out, mlp_b_out)
                hidden_alloc += mlp_sop.hidden_size

        # Assign output layer weights
        out_w = flat_sop.w
        cur_dim = 0
        for dep in flat_sop.deps:
            assert dep in residual_alloc, f"Dependency {dep} of output layer has not been assigned a residual stream allocation."
            dep_start, dep_end = residual_alloc[dep]
            model.lm_head.weight[:, dep_start:dep_end] += out_w[:, cur_dim:cur_dim + dep.dim]
            cur_dim += dep.dim

        # Assign token embeddings
        for tok_emb in tok_embs:
            model.embed_tokens.weight.data += torch.tensor(
                tok_emb.compile_weights(residual_alloc=residual_alloc, embed_dim=embed_dim, tokens=vocab))

        # Assign positional encodings
        for pos_enc in pos_encs:
            model.positional_encoding += torch.tensor(
                pos_enc.compile_weights(residual_alloc=residual_alloc, embed_dim=embed_dim, max_seq_len=max_seq_len))
    if return_alloc:
        return model, residual_alloc, flat_sop
    else:
        return model
