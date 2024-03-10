import os
import time
import math
import torch

from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList, GenerationConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from utils import line_sat, load_model_and_tokenizer, SATStoppingCriteria, is_old_tokenizer, load_conf_file
from eval import batch_generate_completions

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
mini_batch_size =16
temperature = 1.0
gradient_accumulation_steps = 1
epochs = 4
learning_rate = 1.41e-5
lr_scheduler = None
length_penalty = 0.01

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
    device = torch.device("cuda")
    torch.cuda.manual_seed(0)

# To prevent overwriting existing models
if append_timestamp:
    out_dir += f"-{time.strftime('%Y%m%d-%H%M%S')}"

if dataset is None:
    raise ValueError("Please specify a dataset by setting the 'dataset' variable in the config file or using --dataset=[DATASET PATH].")

if model_dir is None:
    raise ValueError("Please specify a model directory by setting the 'model_dir' variable in the config file or using --model_dir=[MODEL DIRECTORY].")

# helper functions for different rewards (consider building classes in a seperate module) for online / oracle
def simple_offline_outcome_supervised_reward(responses, expected):
    # this might be inneficient?
    matches = []
    L_cots = []

    for response, expected_outcome in zip(responses, expected):
        match = 1.0 if (line_sat(response) == line_sat(expected_outcome)) else -1.0
        L_cot = len(response)

        matches.append(match)
        L_cots.append(L_cot)

    m = torch.tensor(matches)
    L_c = torch.tensor(L_cots)

    rewards = m * torch.exp(-length_penalty * L_c)

    # ppo_trainer wants a list of tensors, ffs
    return [reward for reward in rewards]

# Load the model
model, tokenizer = load_model_and_tokenizer(model_dir)
wrapped_model = AutoModelForCausalLMWithValueHead(model)
wrapped_model.is_peft_model = False


# instantiate the dataset
dataset_path = os.path.join(dataset, file_name)
dataset = SATDatasetForRL(
    file_path=dataset_path,
    tokenizer=tokenizer,
    block_size=block_size,
    max_id=max_id,
    permute_constants=permute_constants,
    old_tokenizer=is_old_tokenizer(tokenizer),
)

# set up generation
stop_criteria = SATStoppingCriteria(tokenizer)
stop_criteria = StoppingCriteriaList([stop_criteria]) if stop_crit else None

gen_config = GenerationConfig(
    max_length=min(max_gen_len, model.config.n_positions),
    num_return_sequences=1,
    temperature=temperature,  # not used in eval, could be used here?

)

gen_config.pad_token_id = tokenizer.pad_token_id
gen_config.eos_token_id = tokenizer.eos_token_id

generation_kwargs = {
    "generation_config": gen_config,
    "stopping_criteria": stop_criteria,
    "return_prompt": False,
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
    model = wrapped_model,
    tokenizer = tokenizer,
    dataset = dataset,
    lr_scheduler = lr_scheduler,
)

# Training loop
for epoch in tqdm(range(ppo_trainer.config.ppo_epochs), "epoch: "):
    for batch in tqdm(ppo_trainer.dataloader):
        queries = batch["query"]
        # query_batch = tokenizer.batch_encode_plus(
        #     queries, 
        #     return_tensors="pt", 
        #     padding=True,
        #     truncation=True,
        # )["input_ids"].to(device)
        

        # # It wants it as a list of tensors, ffs
        query_tensors = [tokenizer(query, return_tensors='pt', padding=False)["input_ids"].squeeze().to(device) for query in queries]
        # #### Get response 
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        # response_tensors = ppo_trainer.model.generate(query_batch, **generation_kwargs)
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        # response_tensors = [tokenizer(responses[i].squeeze())["input_ids"] for i in range(len(responses))]
        print(batch["response"][:4])
        print(batch["expected_response"][:4])
        #### Compute reward score
        rewards = simple_offline_outcome_supervised_reward(batch["response"], batch["expected_response"])
        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

# save model
ppo_trainer.save_model(out_dir)