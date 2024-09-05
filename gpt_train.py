import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import (AutoModel, GPT2Config, GPT2LMHeadModel, GPT2Model,
                          LlamaConfig, LlamaForCausalLM, PreTrainedTokenizer,
                          Trainer, TrainingArguments, get_linear_schedule_with_warmup,
                          get_cosine_schedule_with_warmup)
from torch.optim import AdamW
from sat_dataset import SATDataset, CustomTokenizer
from sat_trainer import SATHFTrainer
import utils
import time

### Parameters ###
epochs = 20
batch_size = 12
block_size = 800
max_id = 30
out_dir = 'models/sat-llama'
debug = False
append_timestamp = True
use_wandb = True
remove_trace = False
rand_pos = True
perm_vars = True
load_model = None
old_tokenizer = False
state_trace = True
mask_formula = True
model = "llama"
##################

exec(open('configurator.py').read())

utils.debug = debug
if debug:
    use_wandb = False

# Llama model sizes (up to 7B)
llama_sizes = {
    "llama-70M": {"n_layer": 6, "n_embd": 512, "n_head": 8},
    "llama-160M": {"n_layer": 12, "n_embd": 768, "n_head": 12},
    "llama-410M": {"n_layer": 24, "n_embd": 1024, "n_head": 16},
    "llama-1B": {"n_layer": 24, "n_embd": 2048, "n_head": 32},
    "llama-3B": {"n_layer": 32, "n_embd": 3200, "n_head": 32},
    "llama-7B": {"n_layer": 32, "n_embd": 4096, "n_head": 32},
}

# To prevent overwriting existing models
if append_timestamp:
    out_dir += f"-{time.strftime('%Y%m%d-%H%M%S')}"


# remove | "bar" and UP = [ , BACK 
if state_trace:                                                                                              
    custom_tokens = [str(i) for i in range(30 + 1)] + [str(-i) for i in range(1, 30 + 1)] + ["[SEP]", "SAT", "UNSAT", "[EOS]", "Decide", "[UP]", "D", "[BT]", "[UNK]"]
else:
    custom_tokens = [str(i) for i in range(30 + 1)] + [str(-i) for i in range(1, 30 + 1)] + ["[SEP]", "SAT", "UNSAT", "[EOS]", "[UNK]", "(", ")"]

print("Token Set:", custom_tokens)

# Initialize custom tokenizer
tokenizer = CustomTokenizer(custom_tokens)

# Update the path to the local dataset directory
data_directory = "./datasets/Large_500k_SAT_11_15_marginal_large/"

# Load the training data from the local file
train_file_path = os.path.join(data_directory, "train.txt")

# Custom code to read the local dataset file
with open(train_file_path, 'r') as train_file:
    train_data = train_file.readlines()

# Instantiate the SATDataset
custom_dataset = SATDataset(train_file_path,  # Pass the file path
                            tokenizer=tokenizer,
                            max_id=max_id,
                            block_size=block_size,
                            remove_trace=remove_trace,
                            shift_within_block=rand_pos,
                            permute_constants=perm_vars,
                            mask_formula=mask_formula)

# Split the dataset into training and validation sets
train_size = int(0.9 * len(custom_dataset))
val_size = len(custom_dataset) - train_size
train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])

# Loop over each model size and train
for model_name, model_config in llama_sizes.items():
    print(f"Training {model_name}")
    
    config = LlamaConfig(vocab_size=len(tokenizer.vocab),
                         max_position_embeddings=block_size,
                         num_hidden_layers=model_config["n_layer"],
                         num_attention_heads=model_config["n_head"],
                         hidden_size=model_config["n_embd"],
                         intermediate_size=4 * model_config["n_embd"],
                         bos_token_id=tokenizer.vocab["[EOS]"],
                         eos_token_id=tokenizer.vocab["[EOS]"],
                        )

    model = LlamaForCausalLM(config)

    # Training arguments with evaluation strategy and save strategy
    training_args = TrainingArguments(
        output_dir=f"{out_dir}-{model_name}",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        logging_steps=100,
        save_steps=500,
        eval_steps=500,
        evaluation_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="wandb" if use_wandb else "none",
        run_name=f"{os.path.basename(out_dir)}-{model_name}"
    )

    # Initialize Trainer with train and validation datasets
    trainer = SATHFTrainer(model=model,
                           args=training_args,
                           train_dataset=train_dataset,
                           eval_dataset=val_dataset,
                          )

    # Train the model
    trainer.train()

    # Save the final model and tokenizer
    model.save_pretrained(f"{out_dir}-{model_name}")
    tokenizer.save_pretrained(f"{out_dir}-{model_name}")

print("Training completed for all model sizes.")
