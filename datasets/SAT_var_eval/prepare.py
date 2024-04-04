import os
import urllib.request

# Same directory as this file
cur_path = os.path.dirname(os.path.realpath(__file__))
dataset_folder = "num_var_test"
output_folder = cur_path

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# List of file names to download
file_names = []
for num_var in range(4, 21):
    for condition in ["Random", "Skewed", "Var", "Marginal"]:
        file_name = f"SAT_{num_var}_{condition}_Test.txt"
        file_names.append(file_name)

# Base URL for downloading files
base_url = "https://huggingface.co/datasets/leyanpan/sat-solver/resolve/main/"

# Function to download a file
def download_file(file_name):
    file_url = base_url + dataset_folder + "/" + file_name + "?download=true"
    output_path = os.path.join(output_folder, file_name)
    if not os.path.exists(output_path):
        urllib.request.urlretrieve(file_url, output_path)
        print(f"Downloaded {file_name}")
    else:
        print(f"File {file_name} already exists, skipping.")

# Download files
for file_name in file_names:
    download_file(file_name)
