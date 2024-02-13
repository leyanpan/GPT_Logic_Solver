import torch
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import (AutoModel, 
                          GPT2Config,
                          GPT2LMHeadModel, 
                          GPT2Model,
                          PreTrainedTokenizer, 
                          Trainer, 
                          TrainingArguments, 
                          get_linear_schedule_with_warmup, 
                          get_cosine_schedule_with_warmup)

from torch.optim import AdamW


from sat_dataset import SATDataset, CustomTokenizer
from utils import get_dataset_path, debug_log
import utils
import time

### Custom code block ###

class NanoGPTTrainer(Trainer):
    
    def create_optimizer(self):
        '''
        TODO
        '''
        self.optimizer = AdamW(self.model.parameters(), 
                               lr=6e-4, 
                               betas=(0.9, 0.95), 
                               weight_decay=0.1)
        return self.optimizer

    
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        '''
        TODO
        '''
        
        if optimizer is None:
            optimizer = self.optimizer
            
        self.lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                            num_warmup_steps=2000, 
                                                            num_training_steps=num_training_steps)
        return self.lr_scheduler

    def training_step(self, 
                      model, 
                      inputs):
        '''
        TODO
        '''
        outputs = super().training_step(model, inputs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        return outputs
    
    def save_vocabulary(self, 
                        save_directory, 
                        filename_prefix=None):
        '''
        TODO
        '''
        vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.txt")
        
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token in self.vocab.keys():
                writer.write(token + "\n")

        return (vocab_file,)

### Parameters ###
epochs = 20
batch_size = 12
block_size = 800
max_id = 30
out_dir = 'models/sat-gpt'
dataset = "datasets/SAT_6_10"
debug = False
append_timestamp = True
use_wandb = True
##################

exec(open('configurator.py').read())
utils.debug = debug

# To prevent overwriting existing models
if append_timestamp:
    out_dir += f"-{time.strftime('%Y%m%d-%H%M%S')}"


custom_tokens = [str(i) for i in range(max_id + 1)] + ["-", "[SEP]", "SAT", "UNSAT", "[EOS]", "[UNK]"]

# Initialize custom tokenizer
tokenizer = CustomTokenizer(custom_tokens)
tokenizer.add_special_tokens({'pad_token': '[EOS]'})

# Initialize GPT-2 configuration
config = GPT2Config(vocab_size=len(tokenizer.vocab), 
                    n_ctx=1024,        
                    n_embd=768,        
                    n_layer=12,       
                    n_head=12, 
                    n_positions=block_size,
                    bos_token_id=tokenizer.vocab["[EOS]"],
                    eos_token_id=tokenizer.vocab["[EOS]"],
                )

# Initialize GPT-2 model
model = GPT2LMHeadModel(config)

# Load dataset
dataset_path = get_dataset_path(dataset)
dataset = SATDataset(dataset_path, tokenizer, block_size=block_size)


# Split the dataset into training and validation sets
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Training arguments with evaluation strategy and save strategy
training_args = TrainingArguments(
    output_dir=out_dir,
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
)

# Initialize Trainer with train and validation datasets
trainer = NanoGPTTrainer(model=model,
                         args=training_args,
                         train_dataset=train_dataset,
                         eval_dataset=val_dataset,
                        )

# Train the model
trainer.train()

# Save the final model and tokenizer
model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)
