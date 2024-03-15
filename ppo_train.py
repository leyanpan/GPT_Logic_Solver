import os
import time
import math
import torch

from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList, GenerationConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from utils import line_sat, load_model_and_tokenizer, SATStoppingCriteria, is_old_tokenizer, load_conf_file, pad_max_len
from eval import batch_generate_completions
import argparse

from sat_dataset import CustomTokenizer, SATDatasetForRL 

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning model using PPO")
    parser.add_argument("model_dir", type=str, default=None, help="The path to the model directory.")
    parser.add_argument("dataset", type=str, default=None, help="The path to the training dataset.")
    parser.add_argument("-c", "--config", type=str, default=None, help="The path to the config file.")
    parser.add_argument("-l", "--max_len", type=int, default=850, help="The maximum length of the generated completions.")
    # PPO Training parameters
    parser.add_argument("-b", "--batch_size", type=int, default=128, help="The batch size to use for PPO dataloader.")
    parser.add_argument("-m", "--mini_batch_size", type=int, default=16, help="The mini batch size to use during PPO training.")
    parser.add_argument("-e", "--epochs", type=int, default=4, help="The number of epochs to train for.")
    parser.add_argument("-t", "--temperature", type=float, default=1.0, help="The temperature to use during generation sampling.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1.41e-5, help="The learning rate for training.")
    parser.add_argument("-lp", "--length_penalty", type=float, default=0.01, help="The length penalty to use during PPO training.")
    parser.add_argument("-ga", "--gradient_accumulation_steps", type=int, default=1, help="The number of gradient accumulation steps.")

    parser.add_argument("-b", "--batch_size", type=int, default=128, help="The batch size to use during training.")
    parser.add_argument("-m", "--mini_batch_size", type=int, default=16, help="The mini batch size to use during PPO training.")
    parser.add_argument("-f", "--file_name", type=str, default='train.txt', help="The name of the file in the dataset to use.")
    parser.add_argument("-o", "--out_dir", type=str, default='models/sat-ppo', help="The directory to output models and other training artifacts.")
    parser.add_argument("--cpu", action="store_true", help="Force use of CPU instead of CUDA.")
    parser.add_argument("--stop_crit", action="store_true", help="Use stopping criteria during generation.")
    parser.add_argument("-d", "--debug", action="store_true", help="Print debug information and adjust settings accordingly.")
    parser.add_argument("-nl", "--no_log", action="store_true", help="Don't use Weights & Biases for logging.")
    parser.add_argument("-s", "--seed", type=int, default=0, help="The random seed to use.")
    args = parser.parse_args()
    load_conf_file(args)
    args.use_cuda = not args.cpu and torch.cuda.is_available()
    args.use_wandb = not args.no_log and not args.debug

    return args

# helper functions for different rewards (consider building classes in a seperate module) for online / oracle
def simple_offline_outcome_supervised_reward(responses, expected, length_penalty=0.01):
    # this might be inneficient?
    rewards = []
    # L_cots = []

    for response, expected_outcome in zip(responses, expected):
        L_c = len(response)
        rewards.append(math.exp(-length_penalty * L_c) if (line_sat(response) == line_sat(expected_outcome)) else -1.0)

    return [torch.tensor(reward) for reward in rewards]


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)

    if torch.cuda.is_available() and args.use_cuda:
        device = torch.device("cuda")
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device("cpu")
        use_cuda = False

    # To prevent overwriting existing models
    if args.append_timestamp:
        args.out_dir += f"-{time.strftime('%Y%m%d-%H%M%S')}"

    if args.dataset is None:
        raise ValueError("Please specify a dataset by setting the 'dataset' variable in the config file or using --dataset=[DATASET PATH].")

    if args.model_dir is None:
        raise ValueError("Please specify a model directory by setting the 'model_dir' variable in the config file or using --model_dir=[MODEL DIRECTORY].")


    # Load the model
    model, tokenizer = load_model_and_tokenizer(args.model_dir)
    if args.debug:
        torch.set_printoptions(threshold=5000)
        def hook(module, input, kwargs, output):
            if input[0].shape[1] > 1:
                print(input, kwargs)
        model.transformer.register_forward_hook(hook, with_kwargs=True)
    model = AutoModelForCausalLMWithValueHead(model)
    model.is_peft_model = False
    ref_model = create_reference_model(model)

    # instantiate the dataset
    dataset_path = os.path.join(args.dataset, args.file_name)
    dataset = SATDatasetForRL(
        file_path=dataset_path,
        tokenizer=tokenizer,
        block_size=args.block_size,
        max_id=args.max_id,
        permute_constants=args.permute_constants,
        old_tokenizer=is_old_tokenizer(tokenizer),
    )


    # set up generation
    stop_criteria = SATStoppingCriteria(tokenizer)
    stop_criteria = StoppingCriteriaList([stop_criteria]) if args.stop_crit else None


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
        exp_name=args.exp_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ppo_epochs=args.epochs,
        log_with='wandb' if args.use_wandb else None,
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
        model.save_pretrained(out_dir)
