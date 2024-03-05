import argparse
import torch
from transformers import GPT2LMHeadModel
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sat_dataset import CustomTokenizer  # Assuming sat_dataset.py is in the same directory or in the PYTHONPATH

def get_args():
    parser = argparse.ArgumentParser(description="Compare GPT-2 LM Head models and custom tokenizers to extract and compare token embeddings.")
    parser.add_argument("model_dirs", type=str, nargs='+', help="Directories where the models and tokenizers are stored, separated by spaces.")
    parser.add_argument("--ntokens", type=int, default=30, help="Number of tokens to use for comparison.")
    return parser.parse_args()

def load_model_and_tokenizer(model_dir):
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = CustomTokenizer.from_pretrained(model_dir)
    return model, tokenizer

def get_token_embeddings(model, tokenizer, tokens):
    # Convert list of tokens to tensor of token ids
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_ids_tensor = torch.tensor([token_ids])
    
    # Get embeddings
    with torch.no_grad():
        outputs = model.transformer.wte(token_ids_tensor)  # Use the word token embedding layer of the transformer
    return outputs

def compute_pairwise_cosine_similarity(embeddings):
    # Normalize embeddings to unit length
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    
    # Compute cosine similarity matrix
    cosine_sim_matrix = torch.mm(embeddings_norm, embeddings_norm.transpose(0, 1))
    
    # Extract upper triangle indices excluding the diagonal to avoid comparing an embedding with itself
    triu_indices = torch.triu_indices(cosine_sim_matrix.size(0), cosine_sim_matrix.size(1), offset=1)
    
    # Extract the cosine similarities based on the upper triangle indices
    pairwise_cosine_similarities = cosine_sim_matrix[triu_indices[0], triu_indices[1]]
    
    return pairwise_cosine_similarities

def plot_combined_histogram(cosine_similarities_list, labels, bins=50):
    plt.figure(figsize=(12, 8))
    for i, cosine_similarities in enumerate(cosine_similarities_list):
        plt.hist(cosine_similarities, bins=bins, alpha=0.5, label=f'Model {labels[i]}', edgecolor='k')
    plt.title('Comparison of Pairwise Cosine Similarities Across Models')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

def main():
    args = get_args()
    all_cosine_similarities = []  # List to store all cosine similarities from all models

    for model_dir in args.model_dirs:
        model, tokenizer = load_model_and_tokenizer(model_dir)
        tokens = [str(i) for i in range(1, args.ntokens)]  # Example tokens

        token_embeddings = get_token_embeddings(model, tokenizer, tokens)[0]
        pairwise_cosine_similarities = compute_pairwise_cosine_similarity(token_embeddings)
        all_cosine_similarities.append(pairwise_cosine_similarities.numpy())  # Store numpy array for plotting

    plot_combined_histogram(all_cosine_similarities, args.model_dirs)

if __name__ == "__main__":
    main()
