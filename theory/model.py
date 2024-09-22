import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict, Tuple, Any

act_map = {
    'relu': F.relu,
    'gelu': F.gelu,
    'silu': F.silu,
}


class GLU(nn.Module):
    def __init__(self, activation='relu'):
        super(GLU, self).__init__()
        self.activation = activation

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * act_map[self.activation](gate)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, hidden_size, num_heads, attn_bias=False):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0, "Hidden size must be divisible by Number of Heads"

        self.head_dim = hidden_size // num_heads

        # Linear layers for projecting the input into Q, K, V
        self.q_proj = nn.Linear(embed_dim, hidden_size, bias=attn_bias)
        self.k_proj = nn.Linear(embed_dim, hidden_size, bias=attn_bias)
        self.v_proj = nn.Linear(embed_dim, hidden_size, bias=attn_bias)

        # Output projection layer
        self.out_proj = nn.Linear(hidden_size, embed_dim, bias=attn_bias)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # Project inputs to query, key, value
        q = self.q_proj(x)  # (batch_size, seq_len, hidden_size)
        k = self.k_proj(x)  # (batch_size, seq_len, hidden_size)
        v = self.v_proj(x)  # (batch_size, seq_len, hidden_size)

        # Reshape and permute for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply causal mask (ensuring no attention to future positions)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e4)

        attn_weights = F.softmax(attn_weights, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attn_output = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_len, head_dim)

        # Concatenate heads
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_len, num_heads, head_dim)
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)  # (batch_size, seq_len, hidden_size)

        # Final linear projection to match embed_dim
        output = self.out_proj(attn_output)  # (batch_size, seq_len, embed_dim)

        return output

    def increment_weights(self, q_proj, k_proj, v_proj, o_proj):
        self.q_proj.weight.data += q_proj
        self.k_proj.weight.data += k_proj
        self.v_proj.weight.data += v_proj
        self.out_proj.weight.data += o_proj


class MLPLayer(nn.Module):
    def __init__(self, embed_dim, mlp_dim, activation='relu'):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, mlp_dim * 2)
        self.glu = GLU(activation=activation)
        self.linear2 = nn.Linear(mlp_dim, embed_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.glu(x)
        x = self.linear2(x)
        return x

    def increment_weights(self, w_in=None, b_in=None, w_out=None, b_out=None):
        """
        Increment the weights and biases of the MLP layers.
        Parameters:
            w_in (torch.Tensor): Increment for the weights of the first linear layer.
            b_in (torch.Tensor): Increment for the biases of the first linear layer.
            w_out (torch.Tensor): Increment for the weights of the second linear layer.
            b_out (torch.Tensor): Increment for the biases of the second linear layer.
        """
        if w_in is not None:
            if self.linear1.weight.size() == w_in.size():
                self.linear1.weight.data += w_in
            else:
                raise ValueError(
                    f"w_in shape {w_in.shape} does not match linear1 weight shape {self.linear1.weight.shape}")

        if b_in is not None:
            if self.linear1.bias.size() == b_in.size():
                self.linear1.bias.data += b_in
            else:
                raise ValueError(f"b_in shape {b_in.shape} does not match linear1 bias shape {self.linear1.bias.shape}")

        if w_out is not None:
            if self.linear2.weight.size() == w_out.size():
                self.linear2.weight.data += w_out
            else:
                raise ValueError(
                    f"w_out shape {w_out.shape} does not match linear2 weight shape {self.linear2.weight.shape}")

        if b_out is not None:
            if self.linear2.bias.size() == b_out.size():
                self.linear2.bias.data += b_out
            else:
                raise ValueError(
                    f"b_out shape {b_out.shape} does not match linear2 bias shape {self.linear2.bias.shape}")


def create_causal_mask(seq_len):
    # Causal mask to prevent attending to future tokens
    mask = torch.tril(torch.ones((seq_len, seq_len))).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    return mask


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, hidden_size, activation='relu', use_layer_norm=False):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, hidden_size, num_heads)
        self.use_layer_norm = use_layer_norm

        if use_layer_norm:
            self.ln1 = nn.LayerNorm(embed_dim)
            self.ln2 = nn.LayerNorm(embed_dim)

        self.mlp = MLPLayer(embed_dim, mlp_dim, activation=activation)

    def forward(self, x, attn_mask=None):
        attn_output = self.attn(x, mask=attn_mask)
        x = x + attn_output
        if self.use_layer_norm:
            x = self.ln1(x)

        mlp_output = self.mlp(x)
        x = x + mlp_output
        if self.use_layer_norm:
            x = self.ln2(x)

        return x


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, mlp_dim, hidden_size, max_seq_len, out_size=None,
                 activation='relu', use_layer_norm=False, vocab: List[str] = None):
        super(TransformerModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.out_size = out_size
        self.use_layer_norm = use_layer_norm
        self.vocab = vocab

        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(max_seq_len, embed_dim))

        self.vocab = vocab
        self.token_to_id = {token: idx for idx, token in enumerate(vocab)} if vocab is not None else None

        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim,
                             num_heads=num_heads,
                             mlp_dim=mlp_dim,
                             hidden_size=hidden_size,
                             activation=activation,
                             use_layer_norm=use_layer_norm)
            for _ in range(num_layers)])

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.final_layer_norm = nn.LayerNorm(embed_dim)

        if out_size is None:
            out_size = vocab_size
        self.lm_head = nn.Linear(embed_dim, out_size, bias=False)

    def forward(self, input_ids, pos_mask=None, residual_alloc: Dict[Any, Tuple[int, int]] = None):
        """
        Args:
            input_ids (torch.Tensor): Tensor of shape (batch_size, seq_length) with token IDs.
            pos_mask (torch.Tensor, optional): Binary mask of shape (batch_size, seq_length), where 1 indicates
                                               the position is not a padding token and 0 indicates padding.
                                               If not provided, assume all positions are non-padding.
            residual_alloc (Dict[str, Tuple[int, int]], optional): Mapping variable names to allocations
                                               in the residual stream as (start, end) tuples. Defaults to None.
        Returns:
            logits (torch.Tensor): Output logits from the model.
            residual_dict (dict, optional): If residual_alloc is provided, returns a dictionary mapping
                                            variable names to residual stream values as numpy arrays.
        """
        batch_size, seq_length = input_ids.size()

        # If pos_mask is not provided, assume all positions are non-padding
        if pos_mask is None:
            pos_mask = torch.ones(batch_size, seq_length, device=input_ids.device).bool()

        # Apply token embeddings
        x = self.embed_tokens(input_ids)  # Shape: (batch_size, seq_length, embed_dim)

        # Compute effective positions for each non-padding token
        effective_positions = (pos_mask.cumsum(dim=1) - 1) * pos_mask  # Only count non-padding tokens

        # Apply positional encodings based on the effective positions
        positional_encodings = self.positional_encoding[:self.max_seq_len, :]  # Get up to max_seq_len positions
        effective_pos_encodings = positional_encodings[effective_positions]
        x = x + effective_pos_encodings

        # Create the standard causal mask (seq_length x seq_length)
        causal_mask = torch.tril(torch.ones(seq_length, seq_length, device=x.device)).bool()

        # Expand the causal mask to include the batch dimension and apply pos_mask
        attention_mask = causal_mask.unsqueeze(0) & pos_mask.unsqueeze(1).unsqueeze(2)

        # Pass through transformer layers with modified attention mask
        for layer in self.layers:
            x = layer(x, attn_mask=attention_mask)

        # Apply layer normalization if required
        if self.use_layer_norm:
            x = self.final_layer_norm(x)

        # Final linear layer to produce logits
        logits = self.lm_head(x)

        residual_dict = {}
        if residual_alloc is not None:
            for var_name, (start, end) in residual_alloc.items():
                # Extract the specified slice (start:end) from the final dimension of the residual stream
                # `x[:, :, start:end]` gets the slice of the residual stream for each token in the batch
                residual_slice = x[:, :, start:end].detach().cpu().numpy()
                residual_dict[var_name] = residual_slice

            return logits, residual_dict

        return logits

    def zero_weights(self):
        # Zero out all weights in the model
        for w in self.parameters():
            w.data.zero_()

    def apply_tokens(self, tokens: List[str], residual_alloc: Dict[Any, Tuple[int, int]] = None) -> torch.Tensor:
        """
        Convert a list of tokens to their corresponding IDs and then apply the model.

        Args:
            tokens (List[str]): List of tokens to be converted to IDs.

        Returns:
            torch.Tensor: The output logits from the model.
        """
        if self.token_to_id is None:
            raise ValueError("Vocabulary is not set. Cannot convert tokens to IDs.")

        # Convert tokens to IDs
        input_ids = torch.tensor([self.token_to_id[token] for token in tokens], dtype=torch.long).unsqueeze(0)

        # Apply the model
        return self.forward(input_ids, residual_alloc=residual_alloc)

    def generate(self, prompt: List[str], max_tokens: int = 600, stop_words: List[str] = ['SAT', 'UNSAT']) -> List[str]:
        """
        Generate tokens based on the given prompt until a stop word is encountered or the maximum number of tokens is reached.

        Args:
            prompt (List[str]): List of tokens to start the generation.
            max_tokens (int): Maximum number of tokens to generate.
            stop_words (List[str]): List of stop words that will terminate the generation.

        Returns:
            List[str]: The generated sequence of tokens.
        """
        if self.token_to_id is None:
            raise ValueError("Vocabulary is not set. Cannot convert tokens to IDs.")

        # Initialize the generated sequence with the prompt
        generated_tokens = prompt[:]

        for _ in range(max_tokens):
            # Convert the current tokens to IDs
            input_ids = torch.tensor([self.token_to_id[token] for token in generated_tokens], dtype=torch.long).unsqueeze(0)

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()

            # Apply the model to get the logits
            logits = self.forward(input_ids)

            # Get the logits of the last token
            last_token_logits = logits[:, -1, :]

            # Convert logits to probabilities (softmax)
            probs = torch.softmax(last_token_logits, dim=-1)

            # Sample the next token from the probability distribution
            next_token_id = torch.argmax(probs, dim=-1).item()

            # Convert the token ID back to the corresponding token
            next_token = self.vocab[next_token_id]

            # Append the next token to the generated sequence
            generated_tokens.append(next_token)

            # Check if the generated token is in the stop words
            if next_token in stop_words:
                break

        return generated_tokens

    def state_generate(self, prompt: List[str], max_tokens: int = 600, stop_words: List[str] = ['SAT', 'UNSAT'],
                 state_separators: List[str] = ['[BT]']) -> List[str]:
        """
        Generate tokens based on the given prompt until a stop word is encountered or the maximum number of tokens is reached.
        Keeps only one full state in the input to the Transformer by removing previous states after new ones are fully generated.
        The original prompt is always kept in the input.

        Args:
            prompt (List[str]): List of tokens to start the generation.
            max_tokens (int): Maximum number of tokens to generate.
            stop_words (List[str]): List of stop words that will terminate the generation.
            state_separators (List[str]): List of tokens that denote the end of a state.

        Returns:
            List[str]: The generated sequence of tokens.
        """
        if self.token_to_id is None:
            raise ValueError("Vocabulary is not set. Cannot convert tokens to IDs.")

        # Keep the original prompt unchanged
        prompt_tokens = prompt[:]
        # Initialize generated tokens (excluding the prompt)
        generated_tokens = []
        # Tokens of the current state being generated
        current_state_tokens = []
        # Counter for state separators generated
        num_state_separators_generated = 0
        sep_id = None

        for _ in range(max_tokens):
            # Prepare the input to the model: prompt + current state tokens
            input_tokens = prompt_tokens + current_state_tokens

            # Convert the current tokens to IDs
            input_ids = torch.tensor([self.token_to_id[token] for token in input_tokens], dtype=torch.long).unsqueeze(0)

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()

            # Apply the model to get the logits
            logits = self.forward(input_ids)

            # Get the logits of the last token
            last_token_logits = logits[:, -1, :]

            # Convert logits to probabilities (softmax)
            probs = torch.softmax(last_token_logits, dim=-1)

            # Choose the next token (here using argmax; you can change this to sampling if desired)
            next_token_id = torch.argmax(probs, dim=-1).item()

            # Convert the token ID back to the corresponding token
            next_token = self.vocab[next_token_id]

            # Append the next token to the generated tokens and current state tokens
            generated_tokens.append(next_token)
            current_state_tokens.append(next_token)

            # Check if the generated token is in the stop words
            if next_token in stop_words:
                break

            # Check if the generated token is a state separator
            if next_token in state_separators:
                num_state_separators_generated += 1

                if num_state_separators_generated >= 2:
                    # A new state has been fully generated
                    # Remove the previous state by resetting current_state_tokens to include only the latest state separator
                    current_state_tokens = current_state_tokens[sep_id:]
                    # Reset the counter since we now have one state separator in current_state_tokens
                    num_state_separators_generated = 1

                sep_id = len(current_state_tokens)

        # Return the full sequence: prompt + generated tokens
        return prompt_tokens + generated_tokens

    def find_max_parameter(self):
        """
        Find the maximum parameter value in the model, excluding the positional encodings.

        Returns:
            float: The maximum parameter value in the model.
        """
        max_param = None
        max_param_name = None

        for name, param in self.named_parameters():
            # Skip the positional encoding parameters
            if "positional_encoding" in name:
                continue

            # Find the maximum value in the current parameter
            param_max = param.data.max().item()

            # Update max_param if it's the largest value found so far
            if max_param is None or param_max > max_param:
                max_param = param_max
                max_param_name = name

        return max_param, max_param_name

    def summary(self):
        """
        Print a summary of the model, including the number of layers, heads, parameters, head dimension, and more.
        """
        print("Transformer Model Summary")
        print("=========================")
        print(f"Vocabulary Size: {self.vocab_size}")
        print(f"Embedding Dimension: {self.embed_dim}")
        print(f"Number of Layers: {self.num_layers}")
        print(f"Number of Heads: {self.num_heads}")
        print(f"Head Dimension: {self.embed_dim // self.num_heads}")
        print(f"MLP Dimension: {self.mlp_dim}")
        print(f"Hidden Size: {self.hidden_size}")
        print(f"Maximum Sequence Length: {self.max_seq_len}")
        print(f"Output Size: {self.out_size if self.out_size is not None else self.vocab_size}")
        print(f"Use Layer Normalization: {'Yes' if self.use_layer_norm else 'No'}")

        # Count total number of parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params}")
        print(f"Trainable Parameters: {trainable_params}")

        # List the number of parameters in each transformer block
        print("\nTransformer Block Details:")
        for i, layer in enumerate(self.layers):
            layer_params = sum(p.numel() for p in layer.parameters())
            print(f"  Layer {i + 1}: {layer_params} parameters")

        print("\nOther Parameters:")
        if self.use_layer_norm:
            norm_params = sum(p.numel() for p in self.final_layer_norm.parameters())
            print(f"  Final LayerNorm: {norm_params} parameters")

        lm_head_params = self.lm_head.weight.numel()
        print(f"  LM Head: {lm_head_params} parameters")
