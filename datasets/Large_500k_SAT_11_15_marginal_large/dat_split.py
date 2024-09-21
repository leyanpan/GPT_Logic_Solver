import os
import random

# Define paths
cur_path = os.path.dirname(os.path.realpath(__file__))
data_file_path = os.path.join(cur_path, "SAT_11_15_random_large.txt")
train_file_path = os.path.join(cur_path, "train.txt")
test_file_path = os.path.join(cur_path, "test.txt")

# Parameters for splitting
test_size = 0.2  
random_seed = 42 

# Load the data
with open(data_file_path, 'r') as file:
    data = file.readlines()

# Shuffle data with a fixed seed for consistency
random.seed(random_seed)
random.shuffle(data)

# Split the data
split_index = int(len(data) * (1 - test_size))
train_data = data[:split_index]
test_data = data[split_index:]

# Save the split data
with open(train_file_path, 'w') as file:
    file.writelines(train_data)

with open(test_file_path, 'w') as file:
    file.writelines(test_data)

print(f"Data split into {len(train_data)} training samples and {len(test_data)} testing samples.")
print(f"Training data saved to: {train_file_path}")
print(f"Testing data saved to: {test_file_path}")
