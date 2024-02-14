import torch
from transformers import GPT2LMHeadModel
import os
from sat_dataset import CustomTokenizer  
import utils  
from sklearn.metrics import (f1_score, 
                             accuracy_score, 
                             precision_score, 
                             recall_score)


### Parameters ###
max_gen_len = 600
temperature = 0.01  
batch_size = 5
dataset = '/home/drdata/llm/models/HF_SAT/datasets/SAT_6_10'
file_name = 'test.txt'
model_dir = '/home/drdata/llm/models/HF_SAT/models/sat-gpt-20240130-141141'
##################

exec(open('configurator.py').read())

input_file = os.path.join(dataset, file_name)

torch.manual_seed(0)
torch.cuda.manual_seed(0)

def line_sat(line, sep=' '):
    if sep + 'UNSAT' in line:
        return False
    elif sep + 'SAT' in line:
        return True
    return None

def load_model_and_tokenizer(model_dir):
    custom_tokens = [str(i) for i in range(31)] + ["-", "[SEP]", "SAT", "UNSAT", "[EOS]", "[UNK]"]
    tokenizer = CustomTokenizer.from_pretrained(model_dir, vocab_list=custom_tokens)
    tokenizer.pad_token = "[EOS]"
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    return model, tokenizer

def batch_generate_completions(input_file, model, tokenizer, batch_size, max_length):
    completions = []
    true_labels = []
    pred_labels = []
    with open(input_file, 'r') as file:
        lines = [line.strip().replace("-", "- ") for line in file.readlines()]
    
    for i in range(0, len(lines), batch_size):
        batch_lines = lines[i:i+batch_size]
        batch_prompts = [line[:line.find("[SEP]") + len("[SEP]")] for line in batch_lines]
        batch_true_labels = [line_sat(line) for line in batch_lines]
        true_labels.extend(batch_true_labels)
        
        # Tokenize the prompts and extract input_ids and attention_mask
        tokenized_outputs = tokenizer(batch_prompts, 
                                      return_tensors="pt", 
                                      padding=True, 
                                      truncation=True,
                                      max_length=max_length)
        
        input_ids = tokenized_outputs["input_ids"]
        attention_mask = tokenized_outputs.get("attention_mask")  
        
        # Generate outputs
        if attention_mask is not None:
            outputs = model.generate(input_ids=input_ids, 
                                     attention_mask=attention_mask, 
                                     max_length=max_length, 
                                     temperature=temperature, 
                                     num_return_sequences=1)
        else:
            outputs = model.generate(input_ids=input_ids, 
                                     max_length=max_length, 
                                     temperature=temperature,
                                     num_return_sequences=1)
        
        for output in outputs:
            completion = tokenizer.decode(output, skip_special_tokens=True)
            completions.append(completion)
            pred_labels.append(line_sat(completion))

    return completions, true_labels, pred_labels

model, tokenizer = load_model_and_tokenizer(model_dir)
completions, true_labels, pred_labels = batch_generate_completions(input_file, 
                                                                   model, 
                                                                   tokenizer, 
                                                                   batch_size=batch_size, 
                                                                   max_length=max_gen_len)

# Output the completions
for completion in completions:
    print(completion)

# Evaluate
if true_labels and pred_labels:
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