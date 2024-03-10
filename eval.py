import torch
from transformers import GPT2LMHeadModel, StoppingCriteria, StoppingCriteriaList, GenerationConfig
import os
from sat_dataset import CustomTokenizer
import argparse
import tqdm
from utils import line_sat, load_model_and_tokenizer, SATStoppingCriteria, is_old_tokenizer, load_conf_file
from sklearn.metrics import (f1_score, 
                             accuracy_score, 
                             precision_score, 
                             recall_score)

def batch_generate_completions(input_file, model, tokenizer, batch_size, max_length, stop_criteria=None, debug=False):
    completions = []
    true_labels = []
    pred_labels = []
    old_tokenizer = is_old_tokenizer(tokenizer)
    with open(input_file, 'r') as file:
        if old_tokenizer:
            lines = [line.strip().replace("-", "- ") for line in file.readlines()]
        else:
            lines = [line.strip() for line in file.readlines()]
    gen_config = GenerationConfig(max_length=min(max_length, model.config.n_positions),
                                  num_return_sequences=1)

    gen_config.pad_token_id = tokenizer.pad_token_id
    gen_config.eos_token_id = tokenizer.pad_token_id
    
    for i in tqdm.tqdm(range(0, len(lines), batch_size)):
        batch_lines = lines[i:i+batch_size]
        batch_prompts = [line[:line.find("[SEP]") + len("[SEP]")] for line in batch_lines]
        batch_true_labels = [line_sat(line) for line in batch_lines]
        true_labels.extend(batch_true_labels)

        
        
        # Tokenize the prompts and extract input_ids and attention_mask
        tokenized_outputs = tokenizer(batch_prompts, 
                                      return_tensors="pt", 
                                      padding=True, 
                                      truncation=True)

        input_ids = tokenized_outputs["input_ids"]
        attention_mask = tokenized_outputs.get("attention_mask") 

        input_ids = input_ids.to(model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)
        # Generate outputs
        if attention_mask is not None:
            outputs = model.generate(input_ids=input_ids, 
                                     attention_mask=attention_mask, 
                                     generation_config=gen_config,
                                     stopping_criteria=stop_criteria)
        else:
            outputs = model.generate(input_ids=input_ids, 
                                     generation_config=gen_config,
                                     stopping_criteria=stop_criteria)
        
        
        for output in outputs:
            completion = tokenizer.decode(output, skip_special_tokens=True)
            completion = completion[:completion.find("SAT") + len("SAT") if "SAT" in completion else -1]
            if debug:
                print(completion)
            completions.append(completion)
            pred_labels.append(line_sat(completion))

    return completions, true_labels, pred_labels

def parse_args():
    parser = argparse.ArgumentParser(description="Script to run GPT-2 model for dataset completion and evaluate SAT/UNSAT prediction accuracy")
    parser.add_argument("model_dir", type=str, default=None, help="The path to the model directory.")
    parser.add_argument("dataset", type=str, default=None, help="The path to the dataset.")
    parser.add_argument("-c", "--config", type=str, default=None, help="The path to the config file.")
    parser.add_argument("-l", "--max_len", type=int, default=850, help="The maximum length of the generated completions.")
    parser.add_argument("-i", "--max_id", type=int, default=30, help="The maximum variable ID in the dataset.")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="The batch size to use during generation.")
    parser.add_argument("-f", "--file_name", type=str, default='test.txt', help="The name of the file in the dataset to evaluate on.")
    parser.add_argument("-o", "--out_file", type=str, default=None, help="The path to the output file to output all generated completions.")
    parser.add_argument("--cpu", action="store_true", help="Use CUDA for generation.")
    parser.add_argument("--stop_crit", action="store_true", help="Use stopping criteria during generation.")
    parser.add_argument("-d", "--debug", action="store_true", help="Print debug information.")
    parser.add_argument("-s", "--seed", type=int, default=0, help="The random seed to use.")
    args = parser.parse_args()
    
    load_conf_file(args)
    args.use_cuda = not args.cpu and torch.cuda.is_available()


if __name__ == "__main__":
    args = parse_args()

    if args.dataset is None:
        raise ValueError("Please specify a dataset by setting the 'dataset' variable in the config file or using --dataset=[DATASET PATH].")
    
    if args.model_dir is None:
        raise ValueError("Please specify a model directory by setting the 'model_dir' variable in the config file or using --model_dir=[MODEL DIRECTORY].")
    
    input_fn = os.path.join(args.dataset, args.file_name)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args.model_dir)

    if args.debug:
        print("Loaded Model", model)
        print("Loaded Tokenizer Vocab", tokenizer.vocab)

    if args.use_cuda:
        model.to("cuda")

    stop_criteria = SATStoppingCriteria(tokenizer)
    stop_criteria = StoppingCriteriaList([stop_criteria]) if args.stop_crit else None
    completions, true_labels, pred_labels = batch_generate_completions(input_fn, 
                                                                    model, 
                                                                    tokenizer, 
                                                                    batch_size=args.batch_size, 
                                                                    max_length=args.max_len,
                                                                    stop_criteria=stop_criteria,
                                                                    debug=args.debug)

    with open(args.out_file, 'w') as file:
        for completion in completions:
            file.write(completion + "\n")

    # Evaluate
    if true_labels and pred_labels:
        for i in range(len(true_labels)):
            if pred_labels[i] is None:
                pred_labels[i] = not true_labels[i] # If the model didn't predict anything, it's wrong
        f1 = f1_score(true_labels, pred_labels, pos_label=False)
        acc = accuracy_score(true_labels, pred_labels)
        prec = precision_score(true_labels, pred_labels, pos_label=False)
        recall = recall_score(true_labels, pred_labels, pos_label=False)
        print(f"F1 Score: {f1}")
        print(f"Accuracy: {acc}")
        print(f"Precision: {prec}")
        print(f"Recall: {recall}")
    else:
        print("No labels to evaluate.")