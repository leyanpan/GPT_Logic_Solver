import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizer
from sat_dataset import CustomTokenizer
import utils
from utils import debug_log
import os

### Parameters ###
max_gen_len = 600
max_id = 30
temperature = 0.01 # Low temperature since we're doing formal reasoning
num_samples = 100
max_id = 30
eval = True
debug = True
seed = 0
dataset = 'datasets/SAT_6_10'
file_name = 'test.txt'
model_dir = 'models/sat-6-10-Jan24'  # The directory where your model is saved
##################


exec(open('configurator.py').read())
custom_tokens = [str(i) for i in range(max_id + 1)] + ["-", "[SEP]", "SAT", "UNSAT", "[EOS]", "[UNK]"]

input_file = os.path.join(dataset, file_name)

utils.debug = True
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def line_sat(line, sep=' '):
    assert not ((sep + 'UNSAT') in line and (sep + ' SAT') in line) or sep == ''
    if sep + 'UNSAT' in line:
        return False
    elif sep + 'SAT' in line:
        return True
    else:
        return None

def load_model_and_tokenizer(model_dir):
    # Load the tokenizer
    tokenizer = CustomTokenizer.from_pretrained(model_dir, vocab_list=custom_tokens)

    # Explicitly set pad token ID
    tokenizer.pad_token = "[EOS]"

    # Load the model
    model = GPT2LMHeadModel.from_pretrained(model_dir)

    return model, tokenizer

def generate_completions(input_file, model: GPT2LMHeadModel, tokenizer: PreTrainedTokenizer,  max_length=max_gen_len):
    completions = []
    true_labels = []
    pred_labels = []
    cnt = 0
    with open(input_file, 'r') as file:
        for line in file:
            if cnt >= num_samples:
                break
            cnt += 1
            # debug_log(f"Line: {line}")
            prompt = line.strip().replace("-", "- ")
            true_labels.append(line_sat(prompt))
            prompt = prompt[:prompt.find("[SEP]") + len("[SEP]")]
            # debug_log(f"Prompt: {prompt}")

            # Encode the prompt
            inputs = tokenizer.encode(prompt, return_tensors="pt")

            # Generate a completion
            outputs = model.generate(inputs, max_length=max_length, temperature=temperature, num_return_sequences=1)

            # Decode the output
            completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred_labels.append(line_sat(completion))
            completions.append(completion)
            debug_log(f"Completion: {completion}\n")

    return completions, true_labels, pred_labels


# Load the model and tokenizer
model, tokenizer = load_model_and_tokenizer(model_dir)


# Generate completions
completions, true_labels, pred_labels = generate_completions(input_file, model, tokenizer)

# Output the completions
for completion in completions:
    print(completion)

if eval:
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
    f1 = f1_score(true_labels, pred_labels, pos_label=False)
    acc = accuracy_score(true_labels, pred_labels)
    prec = precision_score(true_labels, pred_labels, pos_label=False)
    recall = recall_score(true_labels, pred_labels, pos_label=False)
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    print(f"Recall: {recall}")
