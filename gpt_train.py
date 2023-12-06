import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizer
from transformers import Trainer, TrainingArguments

epochs = 10
batch_size = 8
block_size = 2500
max_id = 30
out_dir = 'models/sat-gpt'

custom_tokens = [str(i) for i in range(max_id + 1)] + ["-", "[SEP]", "SAT", "UNSAT", "[EOS]", "[UNK]"]

# Custom tokenizer with a placeholder list of tokens
class CustomTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_list, **kwargs):
        super().__init__(vocab_file=None, **kwargs)
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
        
        # Return a dictionary
        return {key: val.squeeze() for key, val in tokens.items()}


# Initialize custom tokenizer
tokenizer = CustomTokenizer(custom_tokens)
tokenizer.add_special_tokens({'pad_token': '[EOS]'})

# Initialize GPT-2 configuration
config = GPT2Config(
    vocab_size=len(tokenizer.vocab),
    n_positions=block_size,
    bos_token_id=tokenizer.vocab["[EOS]"],
    eos_token_id=tokenizer.vocab["[EOS]"],
)

# Initialize GPT-2 model
model = GPT2LMHeadModel(config)

# Load dataset
dataset_path = "datasets/SAT_50k_shuf/train.txt"  # Replace with your dataset file path

# Load dataset
dataset = SATDataset(dataset_path, tokenizer, block_size=block_size)


# Split the dataset into training and validation sets
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Training arguments with evaluation strategy and save strategy
training_args = TrainingArguments(
    output_dir=out_dir,
    num_train_epochs=4,
    per_device_train_batch_size=8,
    logging_steps=100,
    save_steps=500,
    eval_steps=500,
    evaluation_strategy="steps",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False
)

# Initialize Trainer with train and validation datasets
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Save the final model and tokenizer
model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)
