# Dataset File Name: train.txt
# Download file from huggingface if file does not exist
import os

# Same directory as this file
cur_path = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(cur_path, "train.txt")

download_url = "https://huggingface.co/datasets/leyanpan/sat-solver/resolve/main/SAT_Dataset_Diff_Shuf_50k_N15_25_Clause.sat?download=true"

if not os.path.exists(file_path):
    print("Downloading dataset...")
    import urllib.request

    urllib.request.urlretrieve(download_url, file_path)
    print("Dataset downloaded.")