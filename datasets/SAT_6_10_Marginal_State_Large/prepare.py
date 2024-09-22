# Dataset File Name: train.txt
# Download file from huggingface if file does not exist
import os

# Same directory as this file
cur_path = os.path.dirname(os.path.realpath(__file__))
train_file_path = os.path.join(cur_path, "train.txt")
test_file_path = os.path.join(cur_path, "test.txt")

train_data_url = "https://huggingface.co/datasets/leyanpan/sat-solver/resolve/main/large-500k/SAT_6_10_marginal_large.txt?download=true"


if not os.path.exists(train_file_path):
    print("Downloading training dataset...")
    import urllib.request

    urllib.request.urlretrieve(train_data_url, train_file_path)
    print("Dataset downloaded.")
