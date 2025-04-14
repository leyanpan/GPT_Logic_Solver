import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.profiler import profile, record_function, ProfilerActivity

from transformers import (AutoModel,
                          GPT2Config,
                          GPT2LMHeadModel,
                          GPT2Model,
                          LlamaConfig,
                          LlamaForCausalLM,
                          PreTrainedTokenizer,
                          Trainer,
                          TrainingArguments,
                          MambaForCausalLM,
                          MambaConfig,
                          get_linear_schedule_with_warmup,
                          get_cosine_schedule_with_warmup)

from torch.optim import AdamW


from sat_dataset import SATDataset, CustomTokenizer
from sat_trainer import SATHFTrainer
from utils import get_dataset_path
import utils
import time
import os

### Parameters ###
epochs = 20
batch_size = 12
block_size = 800
max_id = 30
out_dir = 'models/sat-gpt'
dataset = None
debug = False
append_timestamp = True
use_wandb = True
remove_trace = False
n_layer = 12
n_embd = 768
n_head = 12
rand_pos = True
perm_vars = True
load_model = None
old_tokenizer = False
state_trace = False
mask_formula = True
model = "gpt2"
##################

exec(open('configurator.py').read())

utils.debug = debug
if debug:
    use_wandb = False

if dataset is None:
    raise ValueError("Please specify a dataset by setting the 'dataset' variable in the config file or using --dataset=[DATASET PATH].")

# To prevent overwriting existing models
if append_timestamp:
    out_dir += f"-{time.strftime('%Y%m%d-%H%M%S')}"

if debug:
    out_dir = "temp"

if old_tokenizer:
    custom_tokens = [str(i) for i in range(30 + 1)] + ["-", "[SEP]", "SAT", "UNSAT", "[EOS]", "[UNK]", "(", ")"]
elif state_trace:
    custom_tokens = [str(i) for i in range(30 + 1)] + [str(-i) for i in range(1, 30 + 1)] + ["[SEP]", "SAT", "UNSAT", "[EOS]", "Decide", "[UP]", "D", "[BT]", "[UNK]"]
else:
    custom_tokens = [str(i) for i in range(30 + 1)] + [str(-i) for i in range(1, 30 + 1)] + ["[SEP]", "SAT", "UNSAT", "[EOS]", "[UNK]", "(", ")"]

print("Token Set:", custom_tokens)

# Initialize custom tokenizer
tokenizer = CustomTokenizer(custom_tokens)

if model == "gpt2":
    # Initialize GPT-2 configuration
    config = GPT2Config(vocab_size=len(tokenizer.vocab),
                        n_ctx=block_size,
                        n_embd=n_embd,
                        n_layer=n_layer,
                        n_head=n_head,
                        n_positions=block_size,
                        bos_token_id=tokenizer.vocab["[EOS]"],
                        eos_token_id=tokenizer.vocab["[EOS]"],
                    )

    # Initialize GPT-2 model
    model = GPT2LMHeadModel(config)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"GPT-2 Total Parameters: {total_params}")

    if load_model is not None:
        print(f"Loading pre-trained GPT-2 model from: {load_model}")
        model = GPT2LMHeadModel.from_pretrained(load_model)

elif model == "llama":
    if load_model is not None:
        print(f"Loading pre-trained Llama model from: {load_model}")
        model = LlamaForCausalLM.from_pretrained(load_model)
    else:
        # For RoPE Encoding
        config = LlamaConfig(vocab_size=len(tokenizer.vocab),
                            max_position_embeddings=block_size,
                            num_hidden_layers=n_layer,
                            num_attention_heads=n_head, # Llama uses n_head
                            hidden_size=n_embd,
                            intermediate_size=4 * n_embd, # Common Llama setting
                            bos_token_id=tokenizer.vocab["[EOS]"],
                            eos_token_id=tokenizer.vocab["[EOS]"],
                            # Add pad_token_id if your tokenizer has one, e.g.
                            # pad_token_id=tokenizer.pad_token_id
                            )

        model = LlamaForCausalLM(config)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Llama Total Parameters (Initialized): {total_params}")


# --- Add Mamba Section ---
elif model == "mamba":
    if load_model is not None:
        print(f"Loading pre-trained Mamba model from: {load_model}")
        # Use MambaForCausalLM for loading language models
        model = MambaForCausalLM.from_pretrained(load_model)
        # Note: Parameter count will be inherent to the loaded model
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Mamba Total Parameters (Loaded): {total_params}")
    else:
        # Initialize Mamba configuration
        # Map common parameters: vocab_size, hidden_size (n_embd), num_hidden_layers (n_layer)
        # Use tokenizer's EOS for bos/eos tokens like other examples
        # Keep Mamba-specific params (state_size, expand, conv_kernel, etc.) at defaults
        config = MambaConfig(
            vocab_size=len(tokenizer.vocab),
            hidden_size=n_embd,
            num_hidden_layers=n_layer,
            bos_token_id=tokenizer.vocab["[EOS]"],
            eos_token_id=tokenizer.vocab["[EOS]"],
            # --- Mamba specific defaults (can be overridden if needed) ---
            # state_size=16,  # Default state space dimension
            # expand=2,       # Default expansion factor for intermediate layer
            # conv_kernel=4,  # Default conv kernel size
            # pad_token_id=0, # Default pad token id (adjust if tokenizer differs)
            # use_bias=False, # Default
            # use_conv_bias=True, # Default
            # hidden_act="silu", # Default
            # initializer_range=0.1, # Default
            # time_step_rank="auto", # Default (depends on hidden_size)
            # residual_in_fp32=True, # Default
            # use_cache=True, # Default: Important for efficient generation
            # use_mambapy=False, # Default, False uses the faster CUDA kernel if available
        )

        # Initialize Mamba model for Causal LM
        model = MambaForCausalLM(config)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Mamba Total Parameters (Initialized): {total_params}")

else:
    raise ValueError(f"Unsupported model type: {model}")




# Load dataset
dataset_path = get_dataset_path(dataset)
dataset = SATDataset(file_path=dataset_path,
                    tokenizer=tokenizer,
                    max_id=max_id,
                    block_size=block_size,
                    remove_trace=remove_trace,
                    shift_within_block=rand_pos,
                    permute_constants=perm_vars,
                    mask_formula=mask_formula,
                    old_tokenizer=old_tokenizer)


# Split the dataset into training and validation sets
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Training arguments with evaluation strategy and save strategy
training_args = TrainingArguments(
    output_dir=out_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    logging_steps=200,
    save_steps=5000,
    #eval_steps=500,
    #evaluation_strategy="steps",
    save_total_limit=2,
    #load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    report_to="wandb" if use_wandb else "none",
    run_name=os.path.basename(out_dir)
)

# Initialize Trainer with train and validation datasets
trainer = SATHFTrainer(model=model,
                         args=training_args,
                         train_dataset=train_dataset,
                         eval_dataset=val_dataset,
                        )

# Train the model
if load_model:
    trainer.train(resume_from_checkpoint=load_model)
else:
    trainer.train()

# Save the final model and tokenizer
model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)