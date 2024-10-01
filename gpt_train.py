import os
import torch
from torch.utils.data import random_split
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    TrainingArguments,
)
from sat_dataset import SATDataset, CustomTokenizer
from sat_trainer import SATHFTrainer
import utils
import time
import argparse


### Parameters ###
epochs = 5
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
model_name = None
train_file = None  # Added train_file as a global variable
resume_checkpoint = None
##################

utils.debug = debug
if debug:
    use_wandb = False
    
# Read additional configurations if any
exec(open('configurator.py').read())

# Argument parsing to select specific model size and train_file
parser = argparse.ArgumentParser(description='Train a specific Llama model size.')
parser.add_argument('--model_name', type=str, help='Name of the model to train (e.g., llama-70M).')
parser.add_argument('--train_file', type=str, help='Path to the training file', default=None)  # Add train_file argument
parser.add_argument('--resume_checkpoint', type=str, help='Path to the checkpoint to resume training', default=None)
args = parser.parse_args()


# Override train_file if passed as a command-line argument
if args.train_file:
    train_file = args.train_file

# Updated Llama model sizes based on standard configurations
llama_sizes = {
    "llama-70M": {
        "n_layer": 16,
        "n_embd": 512,
        "n_head": 8,  # 512 / 8 = 64 per head
    },
    "llama-160M": {
        "n_layer": 24,
        "n_embd": 768,
        "n_head": 12,  # 768 / 12 = 64 per head
    },
    "llama-410M": {
        "n_layer": 32,
        "n_embd": 1024,
        "n_head": 16,  # 1024 / 16 = 64 per head
    },
    "llama-1B": {
        "n_layer": 36,
        "n_embd": 1536,
        "n_head": 24,  # 1536 / 24 = 64 per head
    },
    "llama-3B": {
        "n_layer": 44,
        "n_embd": 2048,
        "n_head": 32,  # 2048 / 32 = 64 per head
    },
    "llama-7B": {
        "n_layer": 60,
        "n_embd": 4096,
        "n_head": 64,  # 4096 / 64 = 64 per head
    },
}

# To prevent overwriting existing models
if append_timestamp:
    out_dir += f"-{time.strftime('%Y%m%d-%H%M%S')}"

# Tokenization
if state_trace:
    custom_tokens = (
        [str(i) for i in range(max_id + 1)] +
        [str(-i) for i in range(1, max_id + 1)] +
        ["[SEP]", "SAT", "UNSAT", "[EOS]", "Decide", "[UP]", "D", "[BT]", "[UNK]"]
    )
else:
    custom_tokens = (
        [str(i) for i in range(max_id + 1)] +
        [str(-i) for i in range(1, max_id + 1)] +
        ["[SEP]", "SAT", "UNSAT", "[EOS]", "[UNK]", "(", ")"]
    )

print("Token Set:", custom_tokens)

# Initialize custom tokenizer
tokenizer = CustomTokenizer(custom_tokens)

# Update the path to the local dataset directory
data_directory = "./datasets/Large_500k_SAT_11_15_marginal_large/"

# Load the training data from the local file (use train_file if provided)
if train_file:
    train_file_path = train_file
else:
    train_file_path = os.path.join(data_directory, "train.txt")

# Custom code to read the local dataset file (Optional, since SATDataset might handle it)
# with open(train_file_path, 'r') as train_file:
#     train_data = train_file.readlines()

# Instantiate the SATDataset
custom_dataset = SATDataset(
    train_file_path,  # Pass the file path
    tokenizer=tokenizer,
    max_id=max_id,
    block_size=block_size,
    remove_trace=remove_trace,
    shift_within_block=rand_pos,
    permute_constants=perm_vars,
    mask_formula=mask_formula
)

# Split the dataset into training and validation sets
train_size = int(0.9 * len(custom_dataset))
val_size = len(custom_dataset) - train_size
train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])

# Determine which models to train
if args.model_name:
    if args.model_name not in llama_sizes:
        available_models = ', '.join(llama_sizes.keys())
        raise ValueError(
            f"Model '{args.model_name}' not found in llama_sizes. Available models: {available_models}"
        )
    model_names = [args.model_name]
else:
    model_names = llama_sizes.keys()

# Loop over each model size and train
for model_name in model_names:
    print(f"Training {model_name}")

    model_config = llama_sizes[model_name]

    config = LlamaConfig(
        vocab_size=len(tokenizer.vocab),
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
        # eval_steps=500,
        # evaluation_strategy="steps",
        save_total_limit=2,
        # load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="wandb" if use_wandb else "none",
        run_name=f"{os.path.basename(out_dir)}-{model_name}",
    )

    # Initialize Trainer with train and validation datasets
    trainer = SATHFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    if args.resume_checkpoint:
        print(f"Resuming training from checkpoint: {args.resume_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_checkpoint)
    else:
        trainer.train()
    
    # Save the final model and tokenizer
    model.save_pretrained(f"{out_dir}-{model_name}")
    tokenizer.save_pretrained(f"{out_dir}-{model_name}")

