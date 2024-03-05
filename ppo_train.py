import os
import time
import math
import torch

from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList, GenerationConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

from sat_dataset import CustomTokenizer, SATDatasetForRL 
# TODO: USE "if __name__ == '__main__':" and argparser in these modules!!
# The way they are now, I can't import these useful things from eval.py!! >:(
# from eval import SATStoppingCriteria, load_model_and_tokenizer, line_sat

### Parameters ###
# generation
max_gen_len = 850
# temperature = 0.01 -- not used in eval; could bn used here?
stop_crit = False

# dataset & tokenizer
max_id = 30
old_tokenizer = False
permute_constants = False
dataset = None
file_name = 'train.txt'

# PPO training
exp_name = dataset
mini_batch_size = 16
gradient_accumulation_steps = 1
epochs = 4
learning_rate = 1.41e-5
lr_scheduler = None
length_penalty = 0.1

use_wandb = True
out_dir = 'models/sat-ppo'
append_timestamp = True

# model
model_dir = None
block_size=800

use_cuda = True
debug = False
##################

# some initialization
exec(open('configurator.py').read())

torch.manual_seed(0)

if torch.cuda.is_available() and use_cuda:
    torch.device("cuda")
    torch.cuda.manual_seed(0)

# To prevent overwriting existing models
if append_timestamp:
    out_dir += f"-{time.strftime('%Y%m%d-%H%M%S')}"

if dataset is None:
    raise ValueError("Please specify a dataset by setting the 'dataset' variable in the config file or using --dataset=[DATASET PATH].")

if model_dir is None:
    raise ValueError("Please specify a model directory by setting the 'model_dir' variable in the config file or using --model_dir=[MODEL DIRECTORY].")

if old_tokenizer:
    custom_tokens = [str(i) for i in range(max_id + 1)] + ["-", "[SEP]", "SAT", "UNSAT", "[EOS]", "[UNK]", "(", ")"]
else:
    custom_tokens = [str(i) for i in range(max_id + 1)] + [str(-i) for i in range(1, max_id + 1)] + ["[SEP]", "SAT", "UNSAT", "[EOS]", "[UNK]", "(", ")"]

# DUPLICATING BECAUSE I CAN'T IMPORT
def line_sat(line, sep=' '):
    if sep + 'UNSAT' in line:
        return False
    elif sep + 'SAT' in line:
        return True
    return None

def load_model_and_tokenizer(model_dir):
    tokenizer = CustomTokenizer(custom_tokens, padding_side="left")
    tokenizer.add_special_tokens({'pad_token': '[EOS]'})
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_dir)
    return model, tokenizer

class SATStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_tokens=['SAT', 'UNSAT', '[EOS]']):
        self.stops = [tokenizer.encode(token)[0] for token in stop_tokens]
        super().__init__()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for row in input_ids:
            if not any(stop_id in row for stop_id in self.stops):
                # If any row does not contain a stop token, continue generation
                return False
        # If all rows contain at least one stop token, stop generation
        return True

# helper functions for different rewards (consider building classes in a seperate module) for online / oracle
def simple_offline_outcome_supervised_reward(responses, expected):
    # this might be inneficient?
    rewards = []

    for response, expected_outcome in zip(responses, expected):
        match = (line_sat(response) == line_sat(expected_outcome)) * 1
        L_cot = len(response)

        rewards.append(match * math.exp(-length_penalty * L_cot))

    return rewards

# Load the model
model, tokenizer = load_model_and_tokenizer(model_dir)

# instantiate the dataset
dataset_path = os.path.join(dataset, file_name)
dataset = SATDatasetForRL(
    file_path=dataset_path,
    tokenizer=tokenizer,
    block_size=block_size,
    max_id=max_id,
    permute_constants=permute_constants,
    old_tokenizer=old_tokenizer,
)

# set up generation
stop_criteria = SATStoppingCriteria(tokenizer)
stop_criteria = StoppingCriteriaList([stop_criteria]) if stop_crit else None

gen_config = GenerationConfig(
    max_length=min(max_gen_len, model.config.n_positions),
    num_return_sequences=1,
    # temperature=temperature,  # not used in eval, could be used here?
)

gen_config.pad_token_id = tokenizer.pad_token_id
gen_config.eos_token_id = tokenizer.pad_token_id

generation_kwargs = {
    "generation_config": gen_config,
    "stopping_criteria": stop_criteria,
}

# Instantiate the PPO trainer
ppo_config = PPOConfig(
    exp_name=exp_name,
    learning_rate=learning_rate,
    mini_batch_size=mini_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    ppo_epochs=epochs,
)

if use_wandb:
    ppo_config.wandb = True

ppo_trainer = PPOTrainer(
    config = ppo_config,
    model = model,
    tokenizer = tokenizer,
    dataset = dataset,
    lr_scheduler = lr_scheduler,
)

# Training loop
for epoch in tqdm(range(ppo_trainer.config.ppo_epochs), "epoch: "):
    for batch in tqdm(ppo_trainer.dataloader):
        queries = batch["query"]
        # query_tensors = tokenizer.batch_encode_plus(
        #     queries, 
        #     return_tensors="pt", 
        #     padding="max_length",
        #     max_length=block_size,
        #     truncation=True,
        # )["input_ids"]

        # This is where it's broken -- I don't know what 
        # ppo_trainer.generate wants.  I give it a batch x max_len 
        # tensor, it says it wants a list.  I give it a list, it
        # says it can't create a tensor.  
        # query_tensors = [
        #     tokenizer.encode(
        #         query,
        #         truncation=True,
        #         padding="max_length",
        #         max_length=block_size,
        #         return_tensors="pt",
        #     ) for query in queries
        # ]

        #### Get response 
        # This is going to be slow, since it's not batched,
        # but ppo_trainer.generate has some weird expectations
        # (read the code) -- nope, this doesn't work either
        response_tensors = [
            ppo_trainer.generate(
                tokenizer.encode(query, return_tensors="pt"),
                **generation_kwargs
            ) for query in queries
        ]

        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
    
        #### Compute reward score
        rewards = simple_offline_outcome_supervised_reward(batch["response"], batch["expected_response"])
    
        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

# save model
ppo_trainer.save_model(out_dir)