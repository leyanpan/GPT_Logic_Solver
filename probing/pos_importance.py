import argparse
import torch
from transformers import GPT2LMHeadModel
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sat_dataset import CustomTokenizer
from utils import load_model_and_tokenizer

def get_args():
    parser = argparse.ArgumentParser(description="Compare GPT-2 LM Head models and custom tokenizers to extract and compare token embeddings.")
    parser.add_argument("model_dirs", type=str, nargs='+', help="Directories where the models and tokenizers are stored, separated by spaces.")
    parser.add_argument("file_path", type=str, help="Path to the file containing the SAT dataset.")
    parser.add_argument("-n", "--nsamples", type=int, default=1, help="Number of SAT instances")
    return parser.parse_args()


def corrupt_indices(input_ids:torch.Tensor, max_id, n_corrupt=5):
    """
    Individually change each position to a DIFFERENT random token in [0, max_id) n_corrupt times.
    param input_ids: Tensor of size (sequence_length,)
    param max_id: int, max token id to replace the original token with
    param n_corrupt: int, number of attempts to corrupt each position
    returns corrupted_ids: Tensor of size (sequence_length, n_corrupt, sequence_length)
    where corrupted_ids[i, j] contains the input_ids with the i-th token replaced by a random token at the j-th attempt
    """
    sequence_length = input_ids.size(0)
    # Expand input_ids to match the target shape: (sequence_length, n_corrupt, sequence_length)
    expanded_input_ids = input_ids.unsqueeze(0).unsqueeze(0).expand(sequence_length, n_corrupt, sequence_length)
    
    # Prepare the corrupted ids tensor, initially copying the expanded_input_ids
    corrupted_ids = expanded_input_ids.clone()
    
    # Generate random indices for corruption
    # Avoiding replacing a token with itself by checking and regenerating the random token if necessary
    for i in range(sequence_length):
        for j in range(n_corrupt):
            cur_id = input_ids[i].item()
            random_token = torch.randint(0, max_id, (1,))
            while random_token == cur_id:
                random_token = torch.randint(0, max_id, (1,))
            corrupted_ids[i, j, i] = random_token
    
    return corrupted_ids

def pos_kl_div_matrix(model: GPT2LMHeadModel, tokenizer: CustomTokenizer, input_text: str, n_corrupt=5):
    # Corrupt the input and get the head vectors for the corrupted input
    if "-1" not in tokenizer.vocab:
        input_text = input_text.replace("-", "- ")
    input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"].to(model.device)
    max_id = model.config.vocab_size
    corrupted_ids = corrupt_indices(input_ids.squeeze(), max_id, n_corrupt)
    sep_token_id = tokenizer.encode("[SEP]")[0]
    sep_token_index = (input_ids[0] == sep_token_id).nonzero(as_tuple=False).item()
    with torch.no_grad():
        print(sep_token_index)
        orig_output = model(input_ids)[0][:, sep_token_index:-1, :]
        len_trace = orig_output.size(1)
        kl_divs = torch.zeros(input_ids.size(-1), len_trace).to(model.device)
        for i in range(corrupted_ids.size(0)):
            batch = corrupted_ids[i]
            corrupted_output = model(batch)[0][:, sep_token_index:-1, :]
            kl_div = F.kl_div(F.log_softmax(corrupted_output, dim=-1), F.log_softmax(orig_output, dim=-1), reduction="none", log_target=True).sum(-1).mean(0)
            kl_divs[i] = kl_div
    return kl_divs

        
def main():
    args = get_args()

    for model_dir in args.model_dirs:
        model, tokenizer = load_model_and_tokenizer(model_dir)
        if torch.cuda.is_available():
            model = model.to("cuda")

        with open(args.file_path, "r") as f:
            for i in range(args.nsamples):
                line = f.readline()
                kl_div = pos_kl_div_matrix(model, tokenizer, line)
                # plot KL matrix
                plt.figure(figsize=(12, 8))
                plt.imshow(kl_div.cpu().numpy().T, cmap="hot", interpolation="nearest")
                plt.colorbar()
                plt.title(f"KL Divergence Matrix for {model_dir}")
                plt.show()

        

if __name__ == "__main__":
    main()