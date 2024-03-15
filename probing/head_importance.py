import argparse
import torch
from transformers import GPT2LMHeadModel
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sat_dataset import CustomTokenizer

def get_args():
    parser = argparse.ArgumentParser(description="Compare GPT-2 LM Head models and custom tokenizers to extract and compare token embeddings.")
    parser.add_argument("model_dirs", type=str, nargs='+', help="Directories where the models and tokenizers are stored, separated by spaces.")
    parser.add_argument("file_path", type=str, help="Path to the file containing the SAT dataset.")
    parser.add_argument("-n", "--nsamples", type=int, default=1, help="Number of SAT instances")
    parser.add_argument("-c", "--corrupt", type=str, choices=["var", "zero"], default="var", help="Corruption strategy to use")
    parser.add_argument("-i", "--index", type=int, default=0, help="Index of the SAT instance to visualize")
    return parser.parse_args()

def load_model_and_tokenizer(model_dir):
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = CustomTokenizer.from_pretrained(model_dir)
    return model, tokenizer

def get_head_vectors(model: GPT2LMHeadModel, input_ids):
    """
    Get all head outputs of an model on a batch of inputs. 
    Output Tensor shape: (batch_size, sequence_length, n_layers, n_heads, head_dim)
    GPT2LMHeadModel has `self.transformer = GPT2Model(config)`
    GPT2Model has `self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])`
    GPT2Block has `self.attn = GPT2Attention(config, layer_idx=layer_idx)`
    GPT2Attention have
    ```
            outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)
    ```
    We should thus get the attn_output from the GPT2Attention module, which is of shape (batch_size, sequence_length, n_heads * head_dim)
    """

    def attention_output_hook(module, input, output):
        # Output is a tuple (attn_output, present), we need attn_output
        module.last_attention_output = output[0]
    
    # Register the hook to the all attention layers
    for layer in model.transformer.h:
        layer.attn.register_forward_hook(attention_output_hook)

    # Forward pass
    outputs = model(input_ids)

    # Get the attention outputs from the hook
    head_vectors = torch.cat([layer.attn.last_attention_output.unsqueeze(2) for layer in model.transformer.h], dim=2)
    # reshape to separate n_heads and head_dim
    head_vectors = head_vectors.view(*head_vectors.shape[:3], model.config.n_head, -1)
    return head_vectors, outputs

def forward_corrupted_head(model: GPT2LMHeadModel, input_ids, corrupted_head_vectors, corrupt_head_idx: list):
    """
    Forward pass with a corrupted head vector.
    corrupted_head_vectors: shape (batch_size, sequence_length, n_layers, n_heads, head_dim)
    corrupt_head_idx: list of indice tuples (layer, head_id) of heads to corrupt
    Returns output logits with head outputs replaced by the corresponding head vectors in corrupted_head_vectors
    """
    hooks = []

    # Function to replace the output of a specific head with the corrupted vector
    def corrupt_head_hook(layer_idx, head_idx, corrupted_vectors):
        def hook(module, input, output):
            # Replace the specified head's output with the corrupted vector
            # Note: output is a tuple (hidden_states, presents, all_attentions) for GPT2LMHeadModel
            # We're interested in modifying the hidden_states, which is output[0]
            hidden_states = output[0]
            # Separate final dimension into n_heads and head_dim, dim is now (batch_size, sequence_length, n_heads, head_dim)
            hidden_states = hidden_states.view(*hidden_states.shape[:2], model.config.n_head, -1)
            # Assign the corrupted vector to the correct position
            hidden_states[:, :, head_idx] = corrupted_vectors[:, :, layer_idx, head_idx]
            # Reshape back to original shape
            hidden_states = hidden_states.view(*hidden_states.shape[:2], -1)
            return (hidden_states,) + output[1:]
        return hook

    # Register hooks based on corrupt_head_idx
    for layer_idx, head_idx in corrupt_head_idx:
        hook = model.transformer.h[layer_idx].attn.register_forward_hook(
            corrupt_head_hook(layer_idx, head_idx, corrupted_head_vectors)
        )
        hooks.append(hook)

    # Perform the forward pass
    with torch.no_grad():
        output_logits = model(input_ids=input_ids)[0] # shape: (batch_size, sequence_length, vocab_size)

    # Cleanup: remove all registered hooks
    for hook in hooks:
        hook.remove()

    return output_logits

def var_corrupt(input_str):
    # Simple strategy: replace all variables (positive or negative) with a random token, negated with 0.5 probability
    tokens = input_str.split()
    corrupted_tokens = []
    for token in tokens:
        if token.isdigit() and token != "0":
            # Replace variable with a random token
            new_token = str(random.randint(1, 10))
            corrupted_tokens.append(new_token)
        elif token.startswith("-") and token[1:].isdigit():
            # Replace negated variable with a random token
            new_token = str(random.randint(1, 10))
            corrupted_tokens.append("-" + new_token)
        else:
            corrupted_tokens.append(token)
    return " ".join(corrupted_tokens)

def zero_corrupt(input_str):
    tokens = input_str.split()
    tokens = ['0' for _ in tokens]
    return " ".join(tokens)

def corrupt_kl_divergence(model, input_ids, corrupt_head_idx, corrupted_head_vectors, original_probs, start):

    # Forward pass with the corrupted head vectors
    output_logits = forward_corrupted_head(model, input_ids, corrupted_head_vectors, corrupt_head_idx)
    original_probs = original_probs
    corrupted_probs = F.log_softmax(output_logits, dim=-1)[:, start:-1, :]
    kl_divergence = F.kl_div(corrupted_probs, original_probs, reduction="none", log_target=True).sum(-1)

    return kl_divergence

def head_kl_div_matrix(model, tokenizer, input_str, corrupt_func=var_corrupt):
    # Corrupt the input and get the head vectors for the corrupted input
    if "-1" not in tokenizer.vocab:
        input_str = input_str.replace("-", "- ")
    corrupted_input = corrupt_func(input_str)
    corrupted_ids = tokenizer(corrupted_input, return_tensors="pt")["input_ids"]
    corrupted_ids = corrupted_ids.to(model.device)
    input_ids = tokenizer(input_str, return_tensors="pt")["input_ids"]
    input_ids = input_ids.to(model.device)

    sep_token_id = tokenizer.encode("[SEP]")[0]
    sep_token_index = (input_ids[0] == sep_token_id).nonzero(as_tuple=False).item()
    # Forward pass with the original input
    with torch.no_grad():
        original_output_logits = model(input_ids=input_ids)[0]

    # Compute KL divergence
    original_probs = F.log_softmax(original_output_logits, dim=-1)[:,sep_token_index:-1,:]
    len_trace = original_probs.shape[1]

    corrupted_head_vectors, _ = get_head_vectors(model, corrupted_ids)
    kl_div_matrix = torch.zeros((model.config.n_layer, model.config.n_head, len_trace)).to(model.device)
    for layer in range(model.config.n_layer):
        for head in range(model.config.n_head):
            kl_div_matrix[layer, head] = corrupt_kl_divergence(model, input_ids, [(layer, head)], corrupted_head_vectors, original_probs, sep_token_index)
    return kl_div_matrix, input_str.split()[sep_token_index + 1:]
    
def main():
    args = get_args()
    corrupt_funcs = {"var": var_corrupt, "zero": zero_corrupt}
    corrupt_func = corrupt_funcs[args.corrupt]

    for model_dir in args.model_dirs:
        model, tokenizer = load_model_and_tokenizer(model_dir)
        if torch.cuda.is_available():
            model = model.to("cuda")

        with open(args.file_path, "r") as f:
            for i in range(args.nsamples):
                line = f.readline()
                kl_div, trace = head_kl_div_matrix(model, tokenizer, line, corrupt_func=corrupt_func)
                plt.figure(figsize=(12, 8))
                plt.imshow(kl_div[:, :, args.index].cpu().numpy(), cmap="hot", interpolation="nearest", vmin=0)
                plt.colorbar()
                plt.title(f"KL Divergence Matrix for {model_dir}, Correct: {trace[args.index]}")
                plt.xlabel("Head Index")
                plt.ylabel("Layer Index")
                plt.show()

        

if __name__ == "__main__":
    main()
