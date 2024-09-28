from typing import List, Union, Tuple, Dict
import numpy as np
import torch.nn.functional as F
import torch
from model import act_map


class SOp:
    deps: List['SOp']
    dim: int

    def __init__(self, deps: List['SOp'], dim: int, name: str = None, **kwargs):
        self.deps = deps
        self.dim = dim
        self.name = name
        for k, v in kwargs.items():
            print(f'Warning: unused kwarg "{k}"')

    def __hash__(self):
        # Include the class name, dim, and deps in the hash
        class_name = self.__class__.__name__
        deps_hash = tuple(self.deps)  # Convert deps list to a tuple for hashing
        return hash((class_name, self.dim, deps_hash))

    def named(self, name: str):
        self.name = name
        return self

    def all_named_deps(self, named_deps: dict | None = None):
        if named_deps is None:
            named_deps = {}
        if self.name:
            named_deps[self.name] = self
        for dep in self.deps:
            if isinstance(dep, SOp):
                dep.all_named_deps(named_deps)
        return named_deps


    def abstract_eval(self, tokens: List[str]):
        """
        Abstract evaluation of the operation.
        Represents the result of the operation without any errors.
        """
        raise NotImplementedError


    def concrete_eval(self, tokens: List[str]):
        """
        Concrete evaluation of the operation. Reduces all operations to SelfAttention, GLUMLP, Linear to perform
        evaluation after eliminating (nearly) all approximation errors.
        """
        raise NotImplementedError

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return MultConst(self, other)
        elif isinstance(other, SOp):
            return MultSOp(self, other)
        else:
            return NotImplementedError

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return AddConst(self, other)
        elif isinstance(other, SOp):
            return AddSOp(self, other)
        else:
            raise NotImplementedError

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return -self + other

    def __gt__(self, other):
        if isinstance(other, (int, float)):
            return GtConst(self, other)
        elif isinstance(other, SOp):
            return GtIntSOp(self, other)
        else:
            raise NotImplementedError

    def __lt__(self, other):
        if isinstance(other, (int, float)):
            return LtConst(self, other)
        elif isinstance(other, SOp):
            return LtIntSOp(self, other)
        else:
            raise NotImplementedError

    def __ge__(self, other):
        if isinstance(other, (int, float)):
            return GeConst(self, other)
        elif isinstance(other, SOp):
            return GeIntSOp(self, other)
        else:
            raise NotImplementedError

    def __le__(self, other):
        if isinstance(other, (int, float)):
            return LeConst(self, other)
        elif isinstance(other, SOp):
            return LeIntSOp(self, other)
        else:
            raise NotImplementedError

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            return EqConst(self, other)
        elif isinstance(other, SOp):
            return EqIntSOp(self, other)
        else:
            raise NotImplementedError

    def __and__(self, other):
        return LogicalAnd(self, other)

    def __getitem__(self, key: Union['SOp', Tuple[Union[slice, 'SOp'], Union[int, slice]]]):
        if isinstance(key, SOp):
            return self.index_by_sop(key)
        elif isinstance(key, tuple) and len(key) == 2:
            # For sop[:, dim_idx] or sop[:, start_dim_idx:end_dim_idx]
            return self.index_by_dim(*key)
        else:
            raise TypeError(f"Invalid index type: {key}")

    def __str__(self):
        return f"<{self.dim}-D SOp {self.name} of type {self.__class__.__name__}>"

    def __repr__(self):
        return self.__str__()

    def index_by_sop(self, index_sop: 'SOp'):
        """
        Simulates token position indexing, i.e., sop[pos_sop] creates a new index_sop such that
        index.abstract_eval(tokens)[i] = sop.abstract_eval(tokens)[pos_sop.abstract_eval(tokens)[i]]
        """
        return IndexBySOp(self, index_sop)

    def index_by_dim(self, pos_sop: Union[slice, 'SOp'], dim_idx: Union[int, slice]):
        if isinstance(dim_idx, int):
            dim_idx = slice(dim_idx, dim_idx + 1)
        elif isinstance(dim_idx, slice):
            if dim_idx.step is not None:
                raise ValueError("Slice step is not supported")
            if dim_idx.start is None:
                dim_idx = slice(0, dim_idx.stop)
            if dim_idx.stop is None:
                dim_idx = slice(dim_idx.start, self.dim)

        if isinstance(pos_sop, slice):
            if pos_sop.step is not None or pos_sop.start is not None or pos_sop.stop is not None:
                raise ValueError("Slice start, stop, and step are not supported as first index")

            return Extract(self, dim_idx.start, dim_idx.stop)
        else:
            return Extract(self[pos_sop], dim_idx.start, dim_idx.stop)

    def sum(self, dim: int=1):
        if dim == 1:
            return SumDims(self)
        elif dim == 0:
            return SumTokens(self)
        else:
            raise ValueError("Invalid dim for sum operation, can only be 0 or 1")


bos_token = '[BOS]'

config = {"mean_exactness": 20}

# Base SOps
class PosEncSOp(SOp):
    def __init__(self, dim, **kwargs):
        super().__init__([], dim, **kwargs)

    def compile_weights(self, residual_alloc: Dict[SOp, Tuple], embed_dim: int, max_seq_len: int):
        empty_tokens = [''] * max_seq_len
        raw_embs = self.abstract_eval(empty_tokens)
        pos_enc_matrix = np.zeros((max_seq_len, embed_dim))
        dim_start, dim_end = residual_alloc[self]
        pos_enc_matrix[:, dim_start:dim_end] += raw_embs

        return pos_enc_matrix

    def concrete_eval(self, tokens: List[str]):
        return self.abstract_eval(tokens)


class Ones(PosEncSOp):
    def __init__(self, **kwargs):
        super().__init__(1, **kwargs)

    def abstract_eval(self, tokens: List[str]):
        return np.ones((len(tokens), 1))


class IsBOS(PosEncSOp):
    def __init__(self, **kwargs):
        super().__init__(1, **kwargs)

    def abstract_eval(self, tokens: List[str]):
        ret = np.zeros((len(tokens), 1))
        ret[0, 0] = 1
        return ret


class ManualArray(PosEncSOp):
    """
    Create a constant positional encoding, usually for test purposes
    """

    def __init__(self, arr: np.ndarray, **kwargs):
        self.token_length = arr.shape[0]
        self.arr = arr
        super().__init__(arr.shape[1], **kwargs)

    def abstract_eval(self, tokens: List[str]):
        assert self.token_length == len(tokens)
        return self.arr


ones = Ones()
is_bos = IsBOS()


class Indices(PosEncSOp):
    def __init__(self, **kwargs):
        super().__init__(1, **kwargs)

    def abstract_eval(self, tokens: List[str]):
        return np.arange(len(tokens)).reshape(-1, 1)


indices = Indices()


class TokEmbSOp(SOp):
    def __init__(self, token_to_id: Dict[str, int], dim: int, **kwargs):
        self.token_to_id = token_to_id
        # List of tokens in the order of the token_to_id dictionary
        self.id_to_token = [None] * len(token_to_id)
        for token, idx in token_to_id.items():
            self.id_to_token[idx] = token
        super().__init__([], dim, **kwargs)

    @property
    def tokens(self):
        return self.id_to_token

    def compile_weights(self, residual_alloc: Dict[SOp, Tuple], embed_dim: int, tokens: List[str]):
        raw_embs = self.abstract_eval(tokens)
        tok_emb_matrix = np.zeros((len(tokens), embed_dim))
        dim_start, dim_end = residual_alloc[self]
        tok_emb_matrix[:, dim_start:dim_end] += raw_embs

        return tok_emb_matrix

    def concrete_eval(self, tokens: List[str]):
        return self.abstract_eval(tokens)


class OneHotTokEmb(TokEmbSOp):
    def __init__(self, token_to_id: Dict[str, int], **kwargs):
        super().__init__(token_to_id, len(token_to_id), **kwargs)

    def abstract_eval(self, tokens: List[str]):
        # One-hot encode the tokens
        return np.eye(len(self.token_to_id))[np.array([self.token_to_id[token] for token in tokens])]


# Linear Transformations that do not require residual allocations or real layers
class Linear(SOp):
    def __init__(self, in_sops: List[SOp] | SOp, w: np.ndarray = None, **kwargs):
        if not isinstance(in_sops, list):
            in_sops = [in_sops]
        self.w = w if w is not None else np.eye(sum(var.dim for var in in_sops))
        super().__init__(in_sops, self.w.shape[0], **kwargs)
        self.check_dims()

    def check_dims(self):
        """
        Check if the dimensions of the provided matrices and variables are correct.
        """
        # Check if w has correct in dimension
        if self.w.shape[1] != sum(var.dim for var in self.deps):
            raise ValueError(f"w should have {sum(var.dim for var in self.deps)} columns, but has {self.w.shape[1]}")

    def _linear_core_eval(self, tokens: List[str], dep_eval_func):
        """
        Core logic for evaluation, can be used by both abstract_eval and concrete_eval.

        Parameters:
        - tokens: List of input tokens.
        - eval_func: Function to evaluate `in_sops` (e.g., abstract_eval or concrete_eval).

        Returns:
        - output: The final evaluated output.
        """
        in_data = np.concatenate([dep_eval_func(var, tokens) for var in self.deps], axis=1)
        return in_data @ self.w.T

    def abstract_eval(self, tokens: List[str]):
        """
        Abstract evaluation of the linear transformation.
        """
        return self._linear_core_eval(tokens, lambda sop, t: sop.abstract_eval(t))

    def concrete_eval(self, tokens: List[str]):
        """
        Concrete evaluation of the linear transformation.
        """
        return self._linear_core_eval(tokens, lambda sop, t: sop.concrete_eval(t))


Id = Linear
Concat = Linear


# Abstract representations of the MLP and self-attention layers
class GLUMLP(SOp):
    def __init__(self, act_sops: Linear | List[SOp] | SOp = None,
                 lin_sops: Linear | List[SOp] | SOp = None,
                 w_lin: np.ndarray = None, b_lin: np.ndarray = None,
                 w_act: np.ndarray = None, b_act: np.ndarray = None,
                 w_out: np.ndarray = None, b_out: np.ndarray = None,
                 activation: str = "relu", **kwargs):
        deps = []
        hidden_size = None
        self.has_act_sops = act_sops is not None
        self.has_lin_sops = lin_sops is not None
        assert self.has_act_sops or self.has_lin_sops, "Either act_sops or lin_sops must be provided"
        if not self.has_act_sops:
            assert w_act is None and b_act is None, ("No activation SOps provided, but weights or biases for "
                                                     "activation are provided.")
        else:
            act_sops = Linear(act_sops, w_act)
            hidden_size = act_sops.dim
            deps += [act_sops]

        if not self.has_lin_sops:
            assert w_lin is None and b_lin is None, ("No linear SOps provided, but weights or biases for linear are "
                                                     "provided.")
        else:
            lin_sops = Linear(lin_sops, w_lin)
            hidden_size = lin_sops.dim
            deps += [lin_sops]

        self.hidden_size = hidden_size
        assert self.hidden_size

        if w_out is None:
            w_out = np.eye(hidden_size)

        self.w_out = w_out
        self.b_out = b_out
        self.b_lin = b_lin
        self.b_act = b_act
        self.activation = activation

        # Initialize in variables and other attributes
        super().__init__(deps, w_out.shape[0], **kwargs)

        # Call the dimension check function
        self.check_dimensions()

    @property
    def act_sops(self) -> Linear | None:
        if not self.has_act_sops:
            return None
        return self.deps[0]

    @property
    def lin_sops(self) -> Linear | None:
        if not self.has_lin_sops:
            return None
        if self.has_act_sops:
            return self.deps[1]
        return self.deps[0]

    def check_dimensions(self):
        """
        Check if the dimensions of the provided matrices and vectors are correct.
        """
        assert self.lin_sops is None or self.lin_sops.dim == self.hidden_size
        assert self.act_sops is None or self.act_sops.dim == self.hidden_size
        assert self.w_out.shape == (self.dim, self.hidden_size)
        assert self.b_out is None or self.b_out.shape == (self.dim,)

    def _glumlp_core_eval(self, tokens: List[str], dep_eval_func):
        """
        Core logic for evaluation, can be used by both abstract_eval and concrete_eval.

        Parameters:
        - tokens: List of input tokens.
        - eval_func: Function to evaluate `act_sops` and `lin_sops` (e.g., abstract_eval or concrete_eval).

        Returns:
        - output: The final evaluated output.
        """

        # Compute the output of the first linear layer
        if self.lin_sops is None:
            lin_output = np.ones((len(tokens), self.hidden_size))
        else:
            lin_output = dep_eval_func(self.lin_sops, tokens)
            if self.b_lin is not None:
                lin_output += self.b_lin

        # Compute the output of the activation layer
        if self.act_sops is None:
            act_output = np.ones((len(tokens), self.hidden_size))
        else:
            act_input = dep_eval_func(self.act_sops, tokens)
            if self.b_act is not None:
                act_input += self.b_act
            act_output = np.maximum(0, act_input)

        # Compute the output of the second linear layer
        output = np.dot((act_output * lin_output), self.w_out.T)
        if self.b_out is not None:
            output += self.b_out

        return output

    def abstract_eval(self, tokens: List[str]):
        """
        Abstract evaluation of the GLUMLP.
        """
        return self._glumlp_core_eval(tokens, lambda sop, t: sop.abstract_eval(t))

    def concrete_eval(self, tokens: List[str]):
        """
        Concrete evaluation of the GLUMLP.
        """
        return self._glumlp_core_eval(tokens, lambda sop, t: sop.concrete_eval(t))

    def compile_weights(self, residual_alloc: Dict[SOp, Tuple], embed_dim: int, mlp_dim: int, hidden_alloc: Tuple):
        """
        Compile the weights for a MLP layer in the transformer model based on the hyperparameters and allocation of SOps.
        Parameters:
            residual_alloc: Dictionary mapping SOps to their residual stream allocation.
            hidden_alloc: Tuple representing the start and end of the allocated hidden dimension.
            mlp_dim: Dimension of the MLP hidden layer of the destination model.
            embed_dim: Dimension of the embedding layer.
        Returns:
            Tuple of four numpy arrays representing the weights for the MLP layer.
        """
        hidden_start, hidden_end = hidden_alloc
        assert hidden_end - hidden_start == self.hidden_size, "GLUMLP Compilation: Allocated hidden dim does not match GLUMLP SOp hidden dim."
        mlp_w_in = np.zeros((2 * mlp_dim, embed_dim))
        mlp_b_in = np.zeros((2 * mlp_dim,))
        mlp_w_out = np.zeros((embed_dim, mlp_dim))
        mlp_b_out = np.zeros((embed_dim,))

        # First half [:mlp_hidden_size] is linear, second half [mlp_hidden_size:] is activation
        # Put the weights in the correct place
        if self.lin_sops is not None:
            cur_dim = 0
            for in_sop in self.lin_sops.deps:
                in_start, in_end = residual_alloc[in_sop]
                mlp_w_in[hidden_start:hidden_end, in_start:in_end] += self.lin_sops.w[:, cur_dim:cur_dim + in_sop.dim]
                cur_dim += in_sop.dim
            if self.b_lin is not None:
                mlp_b_in[hidden_start:hidden_end] += self.b_lin
        else:
            mlp_b_in[hidden_start:hidden_end] += 1

        if self.act_sops is not None:
            cur_dim = 0
            for in_sop in self.act_sops.deps:
                in_start, in_end = residual_alloc[in_sop]
                mlp_w_in[hidden_start + mlp_dim:hidden_end + mlp_dim, in_start:in_end] += self.act_sops.w[:,
                                                                                          cur_dim:cur_dim + in_sop.dim]
                cur_dim += in_sop.dim
            if self.b_act is not None:
                mlp_b_in[hidden_start + mlp_dim:hidden_end + mlp_dim] += self.b_act
        else:
            mlp_b_in[hidden_start + mlp_dim:hidden_end + mlp_dim] += 1

        out_start, out_end = residual_alloc[self]
        mlp_w_out[out_start:out_end, hidden_start:hidden_end] += self.w_out
        if self.b_out is not None:
            mlp_b_out[out_start:out_end] += self.b_out

        return mlp_w_in, mlp_b_in, mlp_w_out, mlp_b_out


class SelfAttention(SOp):
    q: Linear
    k: Linear
    v: Linear
    w_o: np.ndarray

    def __init__(self, q_sops: Linear | List[SOp] | SOp, k_sops: Linear | List[SOp] | SOp,
                 v_sops: Linear | List[SOp] | SOp,
                 w_q: np.ndarray = None, w_k: np.ndarray = None,
                 w_v: np.ndarray = None, w_o: np.ndarray = None, **kwargs):
        # Force q_sops, k_sops, and v_sops to be Linear Operations. This automatically checks dimensions.
        q, k, v = Linear(q_sops, w_q), Linear(k_sops, w_k), Linear(v_sops, w_v)
        # Output weight matrix w_o
        self.w_o = w_o if w_o is not None else np.eye(v.dim)
        # Make sure q_dim = k_dim and v_dim = w_o_dim
        super().__init__([q, k, v], self.w_o.shape[0], **kwargs)
        self.check_dimensions()

    def check_dimensions(self):
        """
        Check if the dimensions of the provided matrices and variables are correct.
        """
        if self.q.dim != self.k.dim:
            raise ValueError(
                f"q_sops have output dimension {self.q.dim} and k_sops have input dimension {self.k.dim}")

    @property
    def q(self) -> Linear:
        return self.deps[0]

    @property
    def k(self) -> Linear:
        return self.deps[1]

    @property
    def v(self) -> Linear:
        return self.deps[2]

    @property
    def hidden_size(self):
        return max(self.q.dim, self.k.dim, self.v.dim)

    def _selfattention_core_eval(self, tokens: List[str], eval_func):
        """
        Core logic for evaluation, can be used by both abstract_eval and concrete_eval.

        Parameters:
        - tokens: List of input tokens.
        - eval_func: Function to evaluate `q`, `k`, and `v` (e.g., abstract_eval or concrete_eval).

        Returns:
        - output: The final evaluated output.
        """

        # Evaluate Q, K, V using the provided eval function
        q, k, v = eval_func(self.q, tokens), eval_func(self.k, tokens), eval_func(self.v, tokens)

        # Compute attention scores (QK^T)
        attention_scores = np.dot(q, k.T)

        # Create causal mask
        causal_mask = np.triu(np.ones_like(attention_scores), k=1) * -1e7

        # Apply causal mask by adding it to attention scores
        attention_scores += causal_mask

        # Apply softmax to get attention weights
        # attention_weights = np.exp(attention_scores - np.max(attention_scores, axis=-1, keepdims=True))
        # attention_weights /= np.sum(attention_weights, axis=-1, keepdims=True)

        attention_scores = torch.tensor(attention_scores)
        attention_weights = F.softmax(attention_scores, dim=-1).numpy()

        # Compute weighted sum of values (attention_weights * V)
        weighted_values = np.dot(attention_weights, v)

        # Apply the output weight matrix w_o
        output = np.dot(weighted_values, self.w_o.T)

        return output

    def abstract_eval(self, tokens: List[str]):
        """
        Abstract evaluation of the self-attention operation.
        """
        return self._selfattention_core_eval(tokens, lambda sop, t: sop.abstract_eval(t))

    def concrete_eval(self, tokens: List[str]):
        """
        Concrete evaluation of the self-attention operation.
        """
        return self._selfattention_core_eval(tokens, lambda sop, t: sop.concrete_eval(t))

    def compile_weights(self, residual_alloc: Dict[SOp, Tuple], embed_dim: int, head_dim: int, head_alloc: int,
                        num_heads: int):
        """
        Compile the weights for a self-attention layer in the transformer model based on the hyperparameters and allocation of SOps.
        Parameters:
            residual_alloc: Dictionary mapping SOps to their residual stream allocation.
            embed_dim: Dimension of the embedding layer/residual stream.
            head_dim: Dimension of the attention head in the destination Transformer.
            head_alloc: The attention head id assigned to the current SOp.
            num_heads: Number of attention heads in the destination Transformer.
        Returns:
            q_proj, k_proj, v_proj, out_proj: Tuple of numpy arrays representing the weights for the self-attention layer.
        """
        hidden_size = head_dim * num_heads

        # Initialize full matrices with zeros
        q_proj = np.zeros((hidden_size, embed_dim))
        k_proj = np.zeros((hidden_size, embed_dim))
        v_proj = np.zeros((hidden_size, embed_dim))
        out_proj = np.zeros((embed_dim, hidden_size))

        # Calculate the start and end indices for the current head within the full matrices
        head_start = head_alloc * head_dim

        # To compensate for the sqrt(d) term in the attention operation Q^T*K
        sqrt4d = head_dim ** 0.25

        # Ensure that the head_dim is large enough for the q, k, v SOps
        assert head_dim >= self.q.dim, f"head_dim ({head_dim}) is smaller than q_sop dim ({self.q.dim})"
        assert head_dim >= self.k.dim, f"head_dim ({head_dim}) is smaller than k_sop dim ({self.k.dim})"
        assert head_dim >= self.v.dim, f"head_dim ({head_dim}) is smaller than v_sop dim ({self.v.dim})"

        # Compile Q weights for the allocated head, using only the first q.dim dimensions
        cur_dim = 0
        for in_sop in self.q.deps:
            in_start, in_end = residual_alloc[in_sop]
            q_proj[head_start:head_start + self.q.dim, in_start:in_end] += sqrt4d * self.q.w[:,
                                                                                    cur_dim:cur_dim + in_sop.dim]
            cur_dim += in_sop.dim

        # Compile K weights for the allocated head, using only the first k.dim dimensions
        cur_dim = 0
        for in_sop in self.k.deps:
            in_start, in_end = residual_alloc[in_sop]
            k_proj[head_start:head_start + self.k.dim, in_start:in_end] += sqrt4d * self.k.w[:,
                                                                                    cur_dim:cur_dim + in_sop.dim]
            cur_dim += in_sop.dim

        # Compile V weights for the allocated head, using only the first v.dim dimensions
        cur_dim = 0
        for in_sop in self.v.deps:
            in_start, in_end = residual_alloc[in_sop]
            v_proj[head_start:head_start + self.v.dim, in_start:in_end] += self.v.w[:, cur_dim:cur_dim + in_sop.dim]
            cur_dim += in_sop.dim

        # Compile output weights, assigning to the appropriate head location in the full matrix
        out_start, out_end = residual_alloc[self]
        out_proj[out_start:out_end, head_start:head_start + self.v.dim] += self.w_o.T

        return q_proj, k_proj, v_proj, out_proj


# Constant SOp by multiplyin ones
class Const(Linear):
    def __init__(self, const: float, ones: Ones = ones, dim: int = 1, **kwargs):
        self.const = const
        super().__init__(ones, np.ones((dim, 1), **kwargs) * const)

    def abstract_eval(self, tokens: List[str]):
        return np.ones((len(tokens), self.dim)) * self.const


# Helper SOp to extract specific dimensions from another SOp
class Extract(Linear):
    def __init__(self, in_var: SOp, start_dim: int, end_dim: int, **kwargs):
        self.in_var = in_var
        self.start_dim = start_dim
        self.end_dim = end_dim

        # Matrix for extracting the desired dimensions
        extract_matrix = np.eye(in_var.dim)[start_dim:end_dim, :]
        super().__init__(in_var, extract_matrix, **kwargs)

    def abstract_eval(self, tokens: List[str]):
        return self.in_var.abstract_eval(tokens)[:, self.start_dim:self.end_dim]


class Pad(Linear):
    def __init__(self, in_sop: SOp, out_dim: int, start_dim: int, **kwargs):
        # Store the input parameters
        self.pad_in_sop = in_sop
        self.pad_start_dim = start_dim

        # Check that the placement of in_sop fits within the output dimensions
        assert start_dim + in_sop.dim <= out_dim, "in_sop's dimensions exceed the output dimensions."

        # Create the weight matrix with zeros everywhere except the placement of in_sop
        w = np.zeros((out_dim, in_sop.dim))
        w[start_dim:start_dim + in_sop.dim, :] = np.eye(in_sop.dim)

        # Initialize the Linear class with in_sop as the dependency and w as the weight matrix
        super().__init__(in_sop, w, **kwargs)

    def abstract_eval(self, tokens: List[str]):
        # Directly create a numpy array with the desired padding
        output = np.zeros((len(tokens), self.dim))  # Initialize with zeros
        in_data = self.pad_in_sop.abstract_eval(tokens)  # Get the evaluation from the input SOp

        # Place in_sop's values at the specified position
        output[:, self.pad_start_dim:self.pad_start_dim + in_data.shape[1]] = in_data

        return output


class BroadCast(Linear):
    def __init__(self, in_var: SOp, dim: int, **kwargs):
        assert in_var.dim == 1, "Only 1-dimensional SOps are broadcastable."
        self.in_var = in_var

        bc_matrix = np.ones((dim, 1))
        super().__init__(in_var, bc_matrix, **kwargs)

    def abstract_eval(self, tokens: List[str]):
        return self.in_var.abstract_eval(tokens).repeat(self.in_var.dim, axis=1)


class MultConst(Linear):
    def __init__(self, in_var: SOp, const: float, **kwargs):
        self.in_var = in_var
        self.const = const

        # Matrix for multiplying by a constant
        mult_matrix = np.eye(in_var.dim) * const
        super().__init__(in_var, mult_matrix, **kwargs)

    def abstract_eval(self, tokens: List[str]):
        return self.in_var.abstract_eval(tokens) * self.const


class AddConst(Linear):
    def __init__(self, in_var: SOp, const: float, ones: SOp = ones, **kwargs):
        self.in_var = in_var
        self.const = const
        self.ones = ones

        ones_vector = np.ones((in_var.dim, 1)) * const
        add_matrix = np.hstack([np.eye(in_var.dim), ones_vector])

        super().__init__([in_var, ones], add_matrix, **kwargs)

    def abstract_eval(self, tokens: List[str]):
        in_data = self.in_var.abstract_eval(tokens)
        return in_data + self.const


class MultSOp(GLUMLP):
    def __init__(self, l_sop: SOp, r_sop: SOp, **kwargs):
        if l_sop.dim == 1 and r_sop.dim != 1:
            l_sop = BroadCast(l_sop, r_sop.dim)

        if r_sop.dim == 1 and l_sop.dim != 1:
            r_sop = BroadCast(r_sop, l_sop.dim)

        self.l_var = l_sop
        self.r_var = r_sop

        assert l_sop.dim == r_sop.dim, (f"Two SOps must have the same dimension, but l_var have {l_sop.dim} and r_var "
                                        f"have {r_sop.dim}")

        # Since act_sops has ReLU activation, we need to use x = ReLU(x) + -ReLU(-x)
        lin_sops = [l_sop, l_sop]
        act_sops = [r_sop, -r_sop]

        w_out = np.hstack([np.eye(l_sop.dim), -np.eye(r_sop.dim)])

        super().__init__(act_sops, lin_sops, w_out=w_out, **kwargs)

    def abstract_eval(self, tokens: List[str]):
        return self.l_var.abstract_eval(tokens) * self.r_var.abstract_eval(tokens)


class AddSOp(Linear):
    def __init__(self, l_sop: SOp, r_sop: SOp, **kwargs):
        if l_sop.dim == 1 and r_sop.dim != 1:
            l_sop = BroadCast(l_sop, r_sop.dim)

        if r_sop.dim == 1 and l_sop.dim != 1:
            r_sop = BroadCast(r_sop, l_sop.dim)

        self.l_sop = l_sop
        self.r_sop = r_sop

        # Ensure the dimensions match for addition
        assert l_sop.dim == r_sop.dim, "Dimensions of l_sop and r_sop must match."

        add_matrix = np.hstack([np.eye(l_sop.dim), np.eye(r_sop.dim)])

        super().__init__([l_sop, r_sop], add_matrix, **kwargs)

    def abstract_eval(self, tokens: List[str]):
        # Perform abstract evaluation by summing the abstract evaluations of l_sop and r_sop
        return self.l_sop.abstract_eval(tokens) + self.r_sop.abstract_eval(tokens)


class GtConst(GLUMLP):
    """
    SOp to determine if in_var > const
    For model_eval, the expected output is:
    1 if in_var >= const + err + 1/2 * eps
    0 if in_var <= const + err - 1/2 * eps
    (in_var - const - err - 1/2 * eps) / eps otherwise
    """

    def __init__(self, in_var: SOp, const: float = 0.0, err=0.5, eps=0.2, **kwargs):
        self.gt_in_var = in_var
        self.gt_const = const
        self.gt_err = err
        self.eps = eps

        zerod_var = in_var - (const + err - 0.5 * eps)
        scaled_var = zerod_var * (1 / eps)
        # ReLU(scaled_var) - ReLU(scaled_var - 1)
        act_sops = [scaled_var, scaled_var - 1]
        w_out = np.hstack([np.eye(scaled_var.dim), -np.eye(scaled_var.dim)])
        super().__init__(act_sops=act_sops, w_out=w_out, **kwargs)

    def abstract_eval(self, tokens: List[str]):
        return self.gt_in_var.abstract_eval(tokens) > self.gt_const + self.gt_err


class GtIntSOp(GtConst):
    def __init__(self, l_var: SOp, r_var: SOp, err=0.5, eps=0.2, **kwargs):
        self.gt_l_var = l_var
        self.gt_r_var = r_var
        zerod_var = (l_var - r_var)
        super().__init__(zerod_var, err=err, eps=eps, **kwargs)

    def abstract_eval(self, tokens: List[str]):
        return self.gt_l_var.abstract_eval(tokens) > self.gt_r_var.abstract_eval(tokens) + self.gt_err


class GeConst(GtConst):
    def __init__(self, in_var: SOp, const: float = 0.0, err=0.5, eps=0.2, **kwargs):
        self.ge_in_var = in_var
        self.ge_const = const
        self.ge_err = err

        super().__init__(in_var, const, -err, eps, **kwargs)

    def abstract_eval(self, tokens: List[str]):
        return self.ge_in_var.abstract_eval(tokens) >= self.ge_const - self.ge_err


class GeIntSOp(GeConst):
    def __init__(self, l_var: SOp, r_var: SOp, err=0.5, eps=0.2, **kwargs):
        self.ge_l_var = l_var
        self.ge_r_var = r_var
        zerod_var = (l_var - r_var)
        super().__init__(zerod_var, err=err, eps=eps, **kwargs)

    def abstract_eval(self, tokens: List[str]):
        return self.ge_l_var.abstract_eval(tokens) >= self.ge_r_var.abstract_eval(tokens) - self.ge_err


class LtConst(GtConst):
    """
    SOp to determine if in_var < const
    For model_eval, the expected output is:
    1 if in_var <= const - err - 1/2 * eps
    0 if in_var >= const - err + 1/2 * eps
    1 - (in_var - const + err - 1/2 * eps) / eps otherwise

    Implementation: -in_var > -const
    """

    def __init__(self, in_var: SOp, const: float = 0.0, err=0.5, eps=0.2, **kwargs):
        self.lt_in_var = in_var
        self.lt_const = const
        self.lt_err = err

        gt_in_var = -self.lt_in_var
        gt_const = -self.lt_const

        super().__init__(gt_in_var, gt_const, err, eps, **kwargs)

    def abstract_eval(self, tokens: List[str]):
        return self.lt_in_var.abstract_eval(tokens) < self.lt_const - self.lt_err


class LtIntSOp(LtConst):
    def __init__(self, l_var: SOp, r_var: SOp, err=0.5, eps=0.2, **kwargs):
        self.lt_l_var = l_var
        self.lt_r_var = r_var
        zerod_var = (l_var - r_var)
        super().__init__(zerod_var, err=err, eps=eps, **kwargs)


class LeConst(GtConst):
    """
    SOp to determine if in_var <= const
    For model_eval, the expected output is:
    1 if in_var <= const + err - 1/2 * eps
    0 if in_var >= const + err + 1/2 * eps
    1 - (in_var - const - err + 1/2 * eps) / eps otherwise

    Implementation: -in_var > -const - err
    """

    def __init__(self, in_var: SOp, const: float = 0.0, err=0.5, eps=0.2, **kwargs):
        self.le_in_var = in_var
        self.le_const = const
        self.le_err = err

        gt_in_var = -self.le_in_var
        gt_const = -self.le_const

        super().__init__(gt_in_var, gt_const, -err, eps, **kwargs)

    def abstract_eval(self, tokens: List[str]):
        return self.le_in_var.abstract_eval(tokens) <= self.le_const + self.le_err


class LeIntSOp(LeConst):
    def __init__(self, l_var: SOp, r_var: SOp, err=0.5, eps=0.2, **kwargs):
        self.le_l_var = l_var
        self.le_r_var = r_var
        zerod_var = (l_var - r_var)
        super().__init__(zerod_var, err=err, eps=eps, **kwargs)

    def abstract_eval(self, tokens: List[str]):
        return self.le_l_var.abstract_eval(tokens) <= self.le_r_var.abstract_eval(tokens) + self.le_err


class EqConst(Linear):
    """
    SOp to simulate if in_var == const, with allowed error
    Expected Output:
    1 if |in_var - const| < err - 1/2 * eps
    0 if |in_var - const| > err + 1/2 * eps
    (|in_var - const| - (err - 1/2 * eps)) / eps otherwise
    """

    def __init__(self, in_var: SOp, const: float = 0, err=0.5, eps=0.2, **kwargs):
        assert err > eps / 2
        self.in_var = in_var
        self.eq_const = const
        self.eq_err = err

        # Implementation: (in_var >= const) and (in_var <= const)
        zerod_var = (in_var - const)
        comp_sum = (GeConst(in_var, const=const, err=err, eps=eps) + LeConst(in_var, const=const, err=err, eps=eps)) - 1

        super().__init__(comp_sum, **kwargs)

    def abstract_eval(self, tokens: List[str]):
        return np.abs(self.in_var.abstract_eval(tokens) - self.eq_const) <= self.eq_err


class EqIntSOp(EqConst):
    def __init__(self, l_var: SOp, r_var: SOp, err=0.5, eps=0.2, **kwargs):
        self.l_var = l_var
        self.r_var = r_var

        zerod_var = (l_var - r_var)
        super().__init__(zerod_var, err=err, eps=eps, **kwargs)


class LogicalAnd(GtConst):
    """
        Simulates b_1 and b_2 between 2 boolean SOps

        Implementation: b_1 + b_2 - 1
    """

    def __init__(self, l_var: SOp, r_var: SOp, **kwargs):
        self.and_l_var = l_var
        self.and_r_var = r_var

        super().__init__(self.and_l_var + self.and_r_var, const=1, **kwargs)

    def abstract_eval(self, tokens: List[str]):
        l_var = self.and_l_var.abstract_eval(tokens)
        r_var = self.and_r_var.abstract_eval(tokens)
        return np.logical_and(l_var, r_var)



# Library functions
class Mean(SelfAttention):
    """
    Find the mean of the values corresponding to the maximum attention logit
    """

    def __init__(self, q_sops: Linear | List[SOp] | SOp, k_sops: Linear | List[SOp] | SOp,
                 v_sops: Linear | List[SOp] | SOp,
                 w_q_mean: np.ndarray = None, w_k_mean: np.ndarray = None,
                 w_v_mean: np.ndarray = None, w_o_mean: np.ndarray = None,
                 bos_weight=None, exactness: float = config["mean_exactness"], eps: float = 1e-3, **kwargs):
        # Store M and eps as class variables
        self.exactness = exactness
        self.eps = eps
        self.mean_bos_weight = bos_weight

        sqrt_ex = np.sqrt(exactness)

        # Incorporate bos_weight directly into self.q_mean and self.k_mean
        if bos_weight is not None:
            self.q_mean = Concat([Linear(q_sops, w_q_mean), Const(bos_weight)])
            self.k_mean = Concat([Linear(k_sops, w_k_mean), is_bos])
        else:
            self.q_mean = Linear(q_sops, w_q_mean)
            self.k_mean = Linear(k_sops, w_k_mean)

        self.v_mean = Linear(v_sops, w_v_mean)
        self.w_o_mean = w_o_mean if w_o_mean is not None else np.eye(self.v_mean.dim)

        q_sa = self.q_mean * sqrt_ex
        k_sa = self.k_mean * sqrt_ex

        # Initialize the SelfAttention superclass with the adjusted weights
        super().__init__(q_sa, k_sa, self.v_mean, w_o=self.w_o_mean, **kwargs)

    def abstract_eval(self, tokens: List[str]):
        """
        Abstract evaluation of the Mean self-attention operation.

        Returns: numpy array of size (num_tokens, v_var_dim)
        """
        q = self.q_mean.abstract_eval(tokens)
        k = self.k_mean.abstract_eval(tokens)
        v = self.v_mean.abstract_eval(tokens)

        attention_logits = np.dot(q, k.T)
        mask = np.tril(np.ones(attention_logits.shape), k=0)
        attention_logits = np.where(mask == 0, -1e9, attention_logits)
        max_attention_logits = np.max(attention_logits, axis=-1)
        max_attention_positions = np.isclose(attention_logits, max_attention_logits[:, None])
        selected_v = np.where(max_attention_positions[:, :, None], v[None, :, :], 0)
        avg_v = np.sum(selected_v, axis=1) / np.sum(max_attention_positions, axis=1, keepdims=True)
        return avg_v


class Copy(Mean):
    """
    Find the last value corresponding to the maximum attention logit
    TODO: Weights are currently too big...
    """

    def __init__(self, q_sops: Linear | List[SOp] | SOp,
                 k_sops: Linear | List[SOp] | SOp,
                 v_sops: Linear | List[SOp] | SOp,
                 indices: SOp = indices, ones: Ones = ones,
                 w_q_mean: np.ndarray = None,
                 w_k_mean: np.ndarray = None,
                 w_v_mean: np.ndarray = None,
                 w_o_mean: np.ndarray = None,
                 bos_weight=None,  # Added bos_weight parameter with default None
                 exactness: float = 10,
                 max_id: float = 100,
                 eps: float = 0.001, **kwargs):

        self.max_id = max_id
        sqrt_M = np.sqrt(max_id)
        self.indices = indices

        # Incorporate bos_weight into copy_q and copy_k, but not copy_v
        if bos_weight is not None:
            self.copy_q = Concat([Linear(q_sops, w_q_mean), Const(bos_weight)])
            self.copy_k = Concat([Linear(k_sops, w_k_mean), is_bos])
        else:
            self.copy_q = Linear(q_sops, w_q_mean)
            self.copy_k = Linear(k_sops, w_k_mean)

        # copy_v remains unchanged
        self.copy_v = Linear(v_sops, w_v_mean)

        # Mean class constructor with bos_weight set to None
        mean_q = Concat([self.copy_q * sqrt_M, ones])
        mean_k = Concat([self.copy_k * sqrt_M, indices])

        super().__init__(mean_q, mean_k, self.copy_v, w_o_mean=w_o_mean, exactness=exactness, eps=eps, bos_weight=None, **kwargs)

    def abstract_eval(self, tokens: List[str]):
        q, k, v, indices = self.copy_q.abstract_eval(tokens), self.copy_k.abstract_eval(
            tokens), self.copy_v.abstract_eval(tokens), self.indices.abstract_eval(tokens)

        # Compute attention logits
        attention_logits = np.dot(q, k.T)

        # Apply causal mask
        causal_mask = np.triu(np.ones_like(attention_logits), k=1) * -1000
        attention_logits += causal_mask

        # Find maximum attention logit and positions within eps
        max_logit = np.max(attention_logits, axis=-1, keepdims=True)
        near_max_positions = np.abs(attention_logits - max_logit) < self.eps  # shape: (num_tokens, num_tokens)

        # Find the last position of the maximum logit
        last_max_position = np.where(near_max_positions, indices.reshape(1, -1), -np.inf).argmax(axis=1)

        # Find the corresponding V vector
        selected_v = v[last_max_position]

        # Apply the output weight matrix w_o_mean
        output = np.dot(selected_v, self.w_o_mean.T)

        return output


class IndexBySOp(Mean):
    def __init__(self, in_sop: Linear | List[SOp] | SOp, index_sop: SOp, indices: SOp = indices, ones: Ones = ones, **kwargs):
        self.in_sop = in_sop
        self.index_sop = index_sop

        assert self.index_sop.dim == 1, f"SOp as an index must have dim == 1, but have dim = {self.index_sop.dim}"

        mean_q = [index_sop * index_sop, index_sop, ones]
        mean_k = [-ones, 2 * indices, -(indices * indices)]

        super().__init__(q_sops=mean_q, k_sops=mean_k, v_sops=in_sop, **kwargs)

    def abstract_eval(self, tokens: List[str]):
        # Evaluate the inputs and indices
        in_arr = self.in_sop.abstract_eval(tokens)
        index_arr = np.round(self.index_sop.abstract_eval(tokens)).reshape(-1).astype(int)

        # Clip each index in the index_arr to the corresponding valid range [0, i]
        clipped_index_arr = np.clip(index_arr, 0, np.arange(len(index_arr)))

        # Return the elements of in_arr at the clipped indices
        return in_arr[clipped_index_arr]


class CPOutput(Linear):
    """
    Simulates the "Conditional Prioritized Output" for token generation, suited for max-prob decoding in a decoder-only transformer model.

    Parameters:
    - vocab_size (int): The size of the vocabulary, determining the dimension of the output vector.
    - conditions_outputs (List[Tuple[SOp | None, int | SOp]]): A list of tuples where each tuple contains:
      - cond (SOp | None): A condition SOp of dimension 1, representing binary variables to determine if the corresponding cond_output should be considered. If None, the cond_output is always considered.
      - cond_output (int | SOp): Either an index representing a specific token in the vocabulary, or an SOp of dimension vocab_size representing a distribution over tokens.

    Behavior:
    The class implements a prioritized conditional output mechanism. For each tuple in the conditions_outputs list, the condition is evaluated in reverse order of priority (earlier in the list = higher priority).

    - If cond is None and cond_output is an index, a one-hot vector for the token is generated and scaled by 2^p, where p is the priority.
    - If cond is None and cond_output is an SOp, the cond_output is directly used with scaling.
    - If cond is an SOp and cond_output is an index, a one-hot vector is generated and multiplied by the condition.
    - If cond is an SOp and cond_output is also an SOp, they are multiplied together to create a new SOp, which is then used.

    The final output is the sum of all the contributions from the conditions and outputs, with each contribution weighted by its priority.
    """

    def __init__(self, vocab_size: int, conditions_outputs: List[Tuple[SOp | None, int | SOp, int]], factor: int = 1, **kwargs):
        deps = []
        w = np.zeros((vocab_size, 0))  # Initialize an empty weight matrix

        for p, (cond, cond_output, weight) in enumerate(reversed(conditions_outputs)):
            priority = weight * factor

            if cond is None:
                if isinstance(cond_output, int):
                    # cond_output is an index, generate a one-hot vector with priority
                    one_hot_vec = np.eye(vocab_size)[:, cond_output:cond_output + 1] * priority
                    w = np.hstack((w, one_hot_vec))
                    deps.append(ones)  # Add a "ones" SOp
                elif isinstance(cond_output, SOp):
                    # Check that cond_output has the correct dimension
                    assert cond_output.dim == vocab_size, "cond_output.dim must equal vocab_size"
                    w = np.hstack((w, np.eye(vocab_size) * priority))
                    deps.append(cond_output)
            else:
                if isinstance(cond_output, int):
                    # cond_output is an index, generate a one-hot vector scaled by cond
                    one_hot_vec = np.eye(vocab_size)[:, cond_output:cond_output + 1] * priority
                    w = np.hstack((w, one_hot_vec))
                    deps.append(cond)
                elif isinstance(cond_output, SOp):
                    # Check that cond_output has the correct dimension
                    assert cond_output.dim == vocab_size, "cond_output.dim must equal vocab_size"
                    # Create a new SOp that is the product of cond and cond_output
                    mult_output = cond * cond_output
                    w = np.hstack((w, np.eye(vocab_size) * priority))
                    deps.append(mult_output)

        super().__init__(deps, w, **kwargs)

    def abstract_eval(self, tokens: List[str]):
        return super().abstract_eval(tokens)

class SumDims(Linear):
    def __init__(self, in_var: SOp):
        super().__init__(in_var, np.ones((1, in_var.dim)))

    def abstract_eval(self, tokens: List[str]):
        return np.sum(self.deps[0].abstract_eval(tokens), axis=1)
