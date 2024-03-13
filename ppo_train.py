import os
import time
import math
import torch

from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList, GenerationConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from utils import line_sat, load_model_and_tokenizer, SATStoppingCriteria, is_old_tokenizer, load_conf_file, pad_max_len
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
batch_size = 128
mini_batch_size = 16
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
if debug:
    use_wandb = False

torch.manual_seed(0)

if torch.cuda.is_available() and use_cuda:
    device = torch.device("cuda")
    torch.cuda.manual_seed(0)
else:
    device = torch.device("cpu")
    use_cuda = False

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
    rewards = []
    # L_cots = []

    for response, expected_outcome in zip(responses, expected):
        L_c = len(response)
        rewards.append(math.exp(-length_penalty * L_c) if (line_sat(response) == line_sat(expected_outcome)) else -1.0)
        # L_cot = len(response)

        # matches.append(match)
        # L_cots.append(L_cot)

    # m = torch.tensor(matches)

    # rewards = m * torch.exp(-length_penalty * L_c)

    # ppo_trainer wants a list of tensors, ffs
    return [torch.tensor(reward) for reward in rewards]

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

# Load the model
model, tokenizer = load_model_and_tokenizer(model_dir)
if debug:
    torch.set_printoptions(threshold=5000)
    def hook(module, input, kwargs, output):
        if input[0].shape[1] > 1:
            print(input, kwargs)
    model.transformer.register_forward_hook(hook, with_kwargs=True)
model = AutoModelForCausalLMWithValueHead(model)
model.is_peft_model = False
ref_model = create_reference_model(model)

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


generation_kwargs = {
    "min_length": -1, # don't ignore the EOS token (see above)
    "top_k": 0.0, # no top-k sampling
    "top_p": 1.0, # no nucleus sampling
    "do_sample": True, # yes, we want to sample
    "pad_token_id": tokenizer.pad_token_id, # most decoder models don't have a padding token - use EOS token instead
    "eos_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 400,
    "return_prompt": False,
}

# Instantiate the PPO trainer
ppo_config = PPOConfig(
    exp_name=exp_name,
    learning_rate=learning_rate,
    batch_size=batch_size,
    mini_batch_size=mini_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    ppo_epochs=epochs,
    log_with='wandb' if use_wandb else None,
)

ppo_trainer = PPOTrainer(
    config = ppo_config,
    model = model,
    ref_model = ref_model,
    tokenizer = tokenizer,
    dataset = dataset,
    data_collator=collator,
)

# Training loop
for epoch in tqdm(range(ppo_trainer.config.ppo_epochs), "epoch: "):
    for batch in tqdm(ppo_trainer.dataloader):
        queries = batch["query"]

        query_tensors = [tokenizer.encode(query, return_tensors='pt', padding=False).squeeze().to(device) for query in queries]
        # #### Get response 
        ppo_trainer.tokenizer.padding_side = "left"
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        # response_tensors = ppo_trainer.model.generate(query_batch, **generation_kwargs)
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        # response_tensors = [tokenizer(responses[i].squeeze())["input_ids"] for i in range(len(responses))]
        #### Compute reward score
        rewards = simple_offline_outcome_supervised_reward(batch["response"], batch["expected_response"])
        # query_tensors = pad_max_len(query_tensors, tokenizer, device)
        #### Run PPO step
        if debug:
            print("Decoded Responses:", batch["response"][:5])
        ppo_trainer.tokenizer.padding_side = "right"
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

# save model
ppo_trainer.save_model(out_dir)