import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

from scipy.stats import linregress

def parse_eval_output(file_path):
    """
    Parses the eval_out.txt file and extracts the evaluation statistics.

    Args:
        file_path (str): Path to the eval_out.txt file.

    Returns:
        List[Dict]: A list of dictionaries containing the extracted data.
    """
    data = []

    with open(file_path, 'r') as f:
        content = f.read()

    # Split the content into blocks for each file
    blocks = content.strip().split('\n\n')

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 8:
            continue  # Skip incomplete blocks

        # Extract the filename
        filename_line = lines[0]
        filename_match = re.match(r'Evaluating file:\s*(.*)', filename_line)
        if not filename_match:
            continue
        filename = filename_match.group(1)

        # Extract number of variables and data type from the filename
        filename_parts = filename.replace('.txt', '').split('_')
        if len(filename_parts) < 3:
            continue  # Invalid filename format
        num_vars = int(filename_parts[1])
        data_type = filename_parts[2]

        # Extract the statistics from the lines
        stats = {}
        for line in lines[1:]:
            key_value = line.split(':')
            if len(key_value) >= 2:
                key = key_value[0].strip()
                value = ':'.join(key_value[1:]).strip()
                stats[key] = value

        # Store the extracted data
        data.append({
            'filename': filename,
            'num_vars': num_vars,
            'data_type': data_type,
            'accuracy': float(stats.get('Accuracy', '0').replace('%', '')),
            'max_c': int(stats.get('Maximum Number of Clauses in Prompt', '0')),
            'max_cot': int(stats.get('Maximum Chain-of-Thought Length', '0')),
            'max_bt': int(stats.get('Maximum Number of Backtracking Steps', '0')),
            'avg_cot': float(stats.get('Average Chain-of-Thought Length', '0')),
            'avg_bt': float(stats.get('Average Number of Backtracking Steps', '0')),
        })

    return data

def create_dataframes(data):
    """
    Creates pandas DataFrames for each data type.

    Args:
        data (List[Dict]): List of dictionaries containing the extracted data.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary mapping data types to their DataFrames.
    """
    df = pd.DataFrame(data)
    dataframes = {}
    for data_type in ['Random', 'Marginal', 'Skewed']:
        df_type = df[df['data_type'].str.lower() == data_type.lower()].copy()
        dataframes[data_type] = df_type.sort_values('num_vars')
    return dataframes

def plot_max_cot(dataframes):
    """
    Plots the scaling of max_cot and avg_cot with respect to the number of variables for each data type.
    The avg_cot is plotted with a more transparent, dashed line using the same color as max_cot.

    Args:
        dataframes (Dict[str, pd.DataFrame]): A dictionary mapping data types to their DataFrames.
    """
    plt.figure(figsize=(10, 6))
    x_values = set()
    color_map = {}  # To store colors assigned to each data type
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']  # Default color cycle
    color_idx = 0  # Index for color cycle

    for data_type, df in dataframes.items():
        x = df['num_vars']
        y_max = df['max_cot']
        y_avg = df['avg_cot']
        x_values.update(x)

        # Assign a color to each data type
        if data_type in color_map:
            color = color_map[data_type]
        else:
            color = color_cycle[color_idx % len(color_cycle)]
            color_map[data_type] = color
            color_idx += 1

        # Plot max_cot
        plt.plot(x, y_max, marker='o', color=color, label=f"{data_type} Max CoT")
        # Plot avg_cot with transparency and dashed line, using the same color
        plt.plot(x, y_avg, marker='o', linestyle='--', color=color, alpha=0.6, label=f"{data_type} Average CoT")

    # Generate the dashed reference lines
    x_min = min(x_values)
    x_max = max(x_values)
    x_ref = np.arange(x_min, x_max + 1)
    y_ref = x_ref * 2 ** (x_ref + 1)
    plt.plot(x_ref, y_ref, linestyle='--', color='gray', label=r'$n \times 2^{p+1}$')
    y_ref_2 = 8 * x_ref * 2 ** (0.08 * x_ref)
    plt.plot(x_ref, y_ref_2, linestyle='--', color='purple', label=r'$8p \times 2^{0.08 p}$')

    plt.yscale('log')
    plt.xlabel('Number of Variables (p)')
    plt.ylabel('Chain-of-Thought Length')
    plt.title('Chain-of-Thought Length vs Number of Variables (p)')
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Path to the eval_out.txt file
    eval_out_path = 'eval_out.txt'  # Adjust this path if necessary

    # Parse the eval_out.txt file
    data = parse_eval_output(eval_out_path)

    # Create pandas DataFrames for each data type
    dataframes = create_dataframes(data)

    # Perform linear regression on transformed data for each data type
    for data_type, df in dataframes.items():
        x = df['num_vars'].values.astype(float)  # n values
        y = df['max_cot'].values.astype(float)  # y values (max_cot)

        # Check if there are enough data points
        if len(x) < 2:
            print(f"Not enough data points to fit for {data_type}.")
            continue

        # Transform the data: compute ln(y / n)
        y_transformed = np.log(y / x)

        # Perform linear regression: y_transformed = c + m * x
        slope, intercept, r_value, p_value, std_err = linregress(x, y_transformed)

        # Compute the parameters a and b
        c = intercept
        m = slope
        a = np.exp(c)
        b = m / np.log(2)  # Since m = b * ln(2)

        print(f"For {data_type}, fitted parameters: a = {a:.6e}, b = {b:.6f}")

    # Print the DataFrames (optional)
    for data_type, df in dataframes.items():
        print(f"\nDataFrame for {data_type}:")
        print(df[['num_vars', 'max_cot']])

    # Plot the scaling of max_cot
    plot_max_cot(dataframes)