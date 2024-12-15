import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt

import os
import re
import pandas as pd

def parse_eval_file(filename, filepath):
    """
    Parses the evaluation file to extract the number of variables, mean exactness beta, accuracy, and test dataset type.

    Returns:
        Tuple containing:
            - num_vars (int): Number of variables.
            - beta (float): Mean Exactness beta.
            - accuracy (float): Accuracy percentage.
            - test_dataset_type (str): Type of test dataset ('Marginal', 'Random', or 'Skewed')
    """
    # Use regex to extract num_vars, type, and beta from the filename
    # Filename format: eval_out_SAT_{num_vars}_{Type}_Test_{beta}.txt
    match = re.match(r'eval_out_SAT_(\d+)_(\w+)_Test_(\d+(\.\d+)?).txt', filename)
    if not match:
        print(f"Filename '{filename}' does not match the expected pattern. Skipping.")
        return None

    num_vars = int(match.group(1))
    dataset_type = match.group(2).capitalize()
    beta = float(match.group(3))

    # Now read the file content to extract the accuracy
    with open(filepath, 'r') as f:
        content = f.read()

    # Use regex to extract accuracy from the content
    accuracy_match = re.search(r'Accuracy:\s*(\d+(\.\d+)?)%', content)
    if not accuracy_match:
        print(f"Could not find accuracy in file '{filename}'. Skipping.")
        return None

    accuracy = float(accuracy_match.group(1))

    return num_vars, beta, accuracy, dataset_type

def parse_compiled_eval_files(directory):
    """
    Parses evaluation files in the specified directory and returns a DataFrame.

    Args:
        directory (str): Directory containing evaluation files.

    Returns:
        DataFrame with columns ['Model', 'Train Dataset', 'Test Dataset', 'num_vars', 'beta', 'accuracy'].
    """
    data = []

    # Iterate over files in the directory
    for filename in os.listdir(directory):
        if not filename.endswith('.txt'):
            continue

        filepath = os.path.join(directory, filename)

        result = parse_eval_file(filename, filepath)
        if result is None:
            continue

        num_vars, beta, accuracy, test_dataset_type = result

        data.append({
            'Model': 'Compiled',
            'Train Dataset': 'Compiled',
            'Test Dataset': test_dataset_type,
            'num_vars': num_vars,
            'beta': beta,
            'accuracy': accuracy
        })

    if not data:
        print("No data collected.")
        return None

    # Create DataFrame
    df = pd.DataFrame(data)
    return df

def main():
    parser = argparse.ArgumentParser(description="Plot accuracy vs number of variables for different Mean Exactness β values.")
    parser.add_argument('directory', type=str, help="Directory containing eval_out files.")
    args = parser.parse_args()

    data = []

    # Iterate over files in the directory
    for filename in os.listdir(args.directory):
        if not filename.endswith('.txt'):
            continue

        filepath = os.path.join(args.directory, filename)

        result = parse_eval_file(filename, filepath)
        if result is None:
            continue

        num_vars, beta, accuracy, _ = result

        data.append({
            'num_vars': num_vars,
            'beta': beta,
            'accuracy': accuracy
        })

    if not data:
        print("No data collected. Exiting.")
        return

    # Create DataFrame
    df = pd.DataFrame(data)

    # Sort by beta and num_vars
    df.sort_values(by=['beta', 'num_vars'], inplace=True)

    # Get list of unique beta values
    beta_values = df['beta'].unique()

    # Plotting
    plt.figure(figsize=(10, 6))

    for beta in beta_values:
        df_beta = df[df['beta'] == beta]
        plt.plot(df_beta['num_vars'], df_beta['accuracy'], marker='o', label=f"β = {beta}")

    plt.xlabel('Number of Variables')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Number of Variables for Different Mean Exactness β Values')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
