import torch
from transformers import GPT2LMHeadModel, StoppingCriteria, StoppingCriteriaList, GenerationConfig
import os
from sat_dataset import CustomTokenizer
import tqdm
import utils  
from sklearn.metrics import (f1_score, 
                             accuracy_score, 
                             precision_score, 
                             recall_score)


### Parameters ###
max_gen_len = 1000
max_id = 30
temperature = 0.01  
batch_size = 24
dataset = None
file_name = 'test.txt'
model_dir = None
out_file = None
use_cuda = True
stop_crit = True
old_tokenizer = False
debug = False
##################


exec(open('configurator.py').read())

if dataset is None:
    raise ValueError("Please specify a dataset by setting the 'dataset' variable in the config file or using --dataset=[DATASET PATH].")

if model_dir is None:
    raise ValueError("Please specify a model directory by setting the 'model_dir' variable in the config file or using --model_dir=[MODEL DIRECTORY].")

if out_file is None:
    out_file = os.path.join('preds', f"{os.path.basename(model_dir)}-{os.path.basename(dataset)}.txt")

if old_tokenizer:
    custom_tokens = [str(i) for i in range(max_id + 1)] + ["-", "[SEP]", "SAT", "UNSAT", "[EOS]", "[UNK]"]
else:
    custom_tokens = [str(i) for i in range(max_id + 1)] + [str(-i) for i in range(1, max_id + 1)] + ["[SEP]", "SAT", "UNSAT", "[EOS]", "[UNK]", "(", ")"]

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

print("Token Set:", custom_tokens)

input_file = os.path.join(dataset, file_name)

# if not os.path.exists(input_file):
#     # run prepare.py in the dataset directory
#     print("Dataset file not found. Running prepare.py in the dataset directory.")
#     cur_dir = os.getcwd()
#     os.chdir(dataset)
#     exec(open('prepare.py').read())
#     os.chdir(cur_dir)

torch.manual_seed(0)
torch.cuda.manual_seed(0)

def line_sat(line, sep=' '):
    if sep + 'UNSAT' in line:
        return False
    elif sep + 'SAT' in line:
        return True
    return None

def load_model_and_tokenizer(model_dir):
    tokenizer = CustomTokenizer(custom_tokens, padding_side="left")
    tokenizer.add_special_tokens({'pad_token': '[EOS]'})
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    return model, tokenizer

def batch_generate_completions(input_file, model, tokenizer, batch_size, max_length, stop_criteria=None):
    completions = []
    true_labels = []
    pred_labels = []
    with open(input_file, 'r') as file:
        if old_tokenizer:
            lines = [line.strip().replace("-", "- ") for line in file.readlines()]
        else:
            lines = [line.strip() for line in file.readlines()]
    gen_config = GenerationConfig(max_new_tokens=max_length,
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

        if use_cuda:
            input_ids = input_ids.to("cuda")
            if attention_mask is not None:
                attention_mask = attention_mask.to("cuda")
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

model, tokenizer = load_model_and_tokenizer(model_dir)

if torch.cuda.is_available() and use_cuda:
    model.to("cuda")
else:
    use_cuda = False

stop_criteria = SATStoppingCriteria(tokenizer)
stop_criteria = StoppingCriteriaList([stop_criteria]) if stop_crit else None
completions, true_labels, pred_labels = batch_generate_completions(input_file, 
                                                                   model, 
                                                                   tokenizer, 
                                                                   batch_size=batch_size, 
                                                                   max_length=max_gen_len,
                                                                   stop_criteria=stop_criteria)

# Output the completions
for completion in completions:
    print(completion)

with open(out_file, 'w') as file:
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