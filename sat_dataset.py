import re
import random
import torch

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

# Custom tokenizer with a placeholder list of tokens
class CustomTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_list, **kwargs):
        super().__init__(**kwargs)
        self.unk_token = "[UNK]"
        self.vocab = {v: k for k, v in enumerate(vocab_list)}
        self.ids_to_tokens = {k: v for v, k in self.vocab.items()}

    def _convert_token_to_id(self, token):
        return self.vocab.get(token)

    def _convert_id_to_token(self, id):
        return self.ids_to_tokens.get(id)

    def tokenize(self, text):
        return text.split()

# Custom dataset class
class SATDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size, max_id, remove_trace=False, shift_within_block=False, permute_constants=False):
        self.tokenizer = tokenizer
        self.max_id = max_id
        self.block_size = block_size
        self.shift_within_block = shift_within_block
        self.permute_constants = permute_constants
        self.examples = []

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip().replace("-", "- ")
                if len(line) > 0:
                    if remove_trace:
                        line = line[:line.find("[SEP]") + len("[SEP]")] + " " + line[line.find("UNSAT") if "UNSAT" in line else line.find("SAT"):]
                    self.examples.append(line)

    def __len__(self):
        return len(self.examples)
    
    def left_pad(self, tensor, pad_size, pad_with, block_size):
        p = torch.tensor([pad_with] * pad_size)

        return torch.cat((p, tensor))[0:block_size]
    
    def multiple_replace(self, string, rep_dict):
        # https://stackoverflow.com/questions/6116978/how-to-replace-multiple-substrings-of-a-string
        # note that iterating over the map causes substrings replaced earler to be replaced again; hence the need for this function
        pattern = re.compile("|".join([re.escape(k) for k in sorted(rep_dict,key=len,reverse=True)]), flags=re.DOTALL)
        return pattern.sub(lambda x: rep_dict[x.group(0)], string)

    def __getitem__(self, i):
        line = self.examples[i]

        if self.permute_constants:
            # If this option is invoked, come up with a random permutation of
            # the constants in the CNF formula and replace them in the line
            # for this SAT problem only
            constants = list(range(1, self.max_id + 1))
            permutation = random.sample(constants, len(constants))

            k = [str(c) + " " for c in constants]
            v = [str(c) + " " for c in permutation]

            p = dict(zip(k, v))

            line = self.multiple_replace(line, p)

        # Tokenize the line
        tokens = self.tokenizer(line, truncation=True, padding='max_length', max_length=self.block_size, return_tensors="pt")

        # Extract input_ids, attention mask as a tensor
        input_ids = tokens["input_ids"].squeeze()
        attention_mask = tokens["attention_mask"].squeeze()

        # Create a label that's the input sequence shifted left by one
        labels = torch.cat((input_ids[1:], torch.tensor([self.tokenizer.pad_token_id])))

        if self.shift_within_block:
            # first get length of unmasked input and determine how much left
            # padding is needed
            input_len = torch.sum(attention_mask)
            size_left_pad = torch.randint(high=(block_size - input_len), size=(1,)).item()

            input_ids = self.left_pad(input_ids, size_left_pad, self.tokenizer.pad_token_id, block_size)
            labels = self.left_pad(labels, size_left_pad, self.tokenizer.pad_token_id, block_size)
            attention_mask = self.left_pad(attention_mask, size_left_pad, 0, block_size)
        
        # Return a dictionary with input_ids and labels
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


if __name__ == "__main__":
    # Test the dataset
    max_id = 30
    custom_tokens = [str(i) for i in range(max_id + 1)] + ["-", "[SEP]", "SAT", "UNSAT", "[EOS]", "[UNK]"]

    tokenizer = CustomTokenizer(custom_tokens)
    tokenizer.add_special_tokens({'pad_token': '[EOS]'})

    block_size = 800

    DATASET_PATH = "./datasets/SAT_6_10/train.txt"
    dataset = SATDataset(
        DATASET_PATH, 
        tokenizer, 
        block_size, 
        max_id,
        remove_trace=False,
        shift_within_block=True,
        permute_constants=True)
    
    # i = torch.randint(0, len(dataset), (1,)).item()
    i = 0
    test_item = dataset.__getitem__(i)
    print(f"Example item, index {i}: \n{test_item}")