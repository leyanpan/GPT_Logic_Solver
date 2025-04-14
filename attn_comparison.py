import torch
import numpy as np
import os
from theory.attn_map import get_logic_attn_maps, post_process_attn_map
from utils import load_model_and_tokenizer, get_num_params, load_conf_file
import argparse

import matplotlib.pyplot as plt


def get_attention_matrix(model, tokenizer, input_str):
    '''
    Get the attention matrix of all the model's heads at each layer for the given input string
    model: A trained SAT-Solving model
    input_str: a string composed of DIMACS encoding of SAT formula and reasoning traces
    Output: A numpy matrix of size [l, h, n, n] where l is the number of layers, h is the number of heads, n is the number of tokens in the sequence
    '''

    # Tokenize input
    encoded_input = tokenizer(
        input_str,
        return_tensors="pt",
        truncation=True
    )

    # Remove token_type_ids if present (not used in GPT models)
    if 'token_type_ids' in encoded_input:
        encoded_input.pop('token_type_ids')

    # Move to the same device as the model
    encoded_input = {k: v.to(model.device) for k, v in encoded_input.items()}

    # Run forward pass with output_attentions=True to get attention matrices
    with torch.no_grad():
        outputs = model(**encoded_input, output_attentions=True)

    # Extract attention matrices from outputs
    attention_matrices = outputs.attentions

    # Convert to numpy array
    attention_numpy = [attn.cpu().numpy() for attn in attention_matrices]

    # Stack the attention matrices into a single numpy array
    # Shape: [layers, batch, heads, seq_len, seq_len]
    stacked_attention = np.stack(attention_numpy)

    # Remove batch dimension (since we only have one input)
    # Final shape: [layers, heads, seq_len, seq_len]
    final_attention = np.squeeze(stacked_attention, axis=1)

    return final_attention

def parse_args():
    parser = argparse.ArgumentParser(description="Script to run GPT-2 model for dataset completion and evaluate SAT/UNSAT prediction accuracy")
    parser.add_argument("model_dir", type=str, default=None, help="The path to the model directory.")
    parser.add_argument("dataset", type=str, default=None, help="The path to the dataset.")
    parser.add_argument("-c", "--config", type=str, default=None, help="The path to the config file.")
    parser.add_argument("-l", "--max_len", type=int, default=850, help="The maximum length of the generated completions.")
    parser.add_argument("-n", "--num_samples", type=int, default=20, help="The number of samples to generate.")
    parser.add_argument("-i", "--max_id", type=int, default=30, help="The maximum variable ID in the dataset.")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="The batch size to use during generation.")
    parser.add_argument("-f", "--file_name", type=str, default='test.txt', help="The name of the file in the dataset to evaluate on.")
    parser.add_argument("-o", "--out_file", type=str, default=None, help="The path to the output file to output all generated completions.")
    parser.add_argument("--cpu", action="store_true", help="Use CUDA for generation.")
    parser.add_argument("--stop_crit", action="store_true", help="Use stopping criteria during generation.")
    parser.add_argument("-d", "--debug", action="store_true", help="Print debug information.")
    parser.add_argument("-s", "--seed", type=int, default=0, help="The random seed to use.")
    args = parser.parse_args()

    load_conf_file(args)
    args.use_cuda = not args.cpu and torch.cuda.is_available()
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.dataset is None:
        raise ValueError("Please specify a dataset by setting the 'dataset' variable in the config file or using --dataset=[DATASET PATH].")

    if args.model_dir is None:
        raise ValueError("Please specify a model directory by setting the 'model_dir' variable in the config file or using --model_dir=[MODEL DIRECTORY].")

    input_fn = os.path.join(args.dataset, args.file_name)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args.model_dir)

    if args.use_cuda:
        model.to("cuda")

    if args.debug:
        print("===== Model Debug Info =====")
        print("Model:", model)  # or model.__class__.__name__
        print("Configuration:", model.config)
        print("Loaded Tokenizer Vocab", tokenizer.vocab)
        print(f" - Number of layers: {model.config.num_hidden_layers}")
        print(f" - Hidden size: {model.config.hidden_size}")
        print(f" - Num attention heads: {model.config.num_attention_heads}")
        print(f" - Vocab size: {model.config.vocab_size}")
        print(f" - Max position embeddings: {model.config.max_position_embeddings}")
        print(f" - Number of Trainable Params: {get_num_params(model):,}")
        for name, param in model.named_parameters():
            print(name, param.shape, param.numel())
        print("===== End Debug Info =====")

    dataset_file_name = os.path.join(args.dataset, args.file_name)
    with open(dataset_file_name, 'r') as f:
        samples = f.readlines()

    for i, sample in enumerate(samples):
        # Get the attention matrix for the sample
        attention_matrix = get_attention_matrix(model, tokenizer, sample)
        tokens = tokenizer.tokenize(sample)
        attention_matrix = post_process_attn_map(attention_matrix, tokens, bos_token=False) # shape: [n_layers, n_heads, n_reasoning_tokens, n_clauses]
        theory_attn_maps = get_logic_attn_maps(tokens) # shape: [n_reasoning_tokens, n_clauses]
        for key, theory_attn_map in theory_attn_maps.items():
            dot_products = np.sum(attention_matrix * theory_attn_map[None, None, :, :], axis=-1)  # shape: [n_layers, n_heads, n_reasoning_tokens]

            # Average the dot product values over the n_reasoning_tokens dimension.
            avg_dot = np.mean(dot_products, axis=-1)  # shape: [n_layers, n_heads]

            # Get the maximum value among all layers and heads.
            max_avg_dot = np.max(avg_dot)

            print(f"Highest average dot product over all layers and heads for {key}:", max_avg_dot)



