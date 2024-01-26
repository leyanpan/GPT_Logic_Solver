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
    def __init__(self, file_path, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = []

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip().replace("-", "- ")
                if len(line) > 0:
                    self.examples.append(line)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Tokenize the line
        tokens = self.tokenizer(self.examples[i], truncation=True, padding='max_length', max_length=self.block_size, return_tensors="pt")
        # Extract input_ids as a tensor
        input_ids = tokens["input_ids"].squeeze()

        # Create labels (which are the same as input_ids for language modeling)
        labels = input_ids.clone()
        
        # Return a dictionary with input_ids and labels
        return {"input_ids": input_ids, "labels": labels}