import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process dataset files and compute accuracies.")
    parser.add_argument('directory', type=str, help="Directory containing dataset files")
    return parser.parse_args()


def parse_filename(filename):
    filename_lower = filename.lower().replace('.txt', '')

    dataset_types = ['marginal', 'random', 'skewed']

    parts = filename_lower.split('_')

    # Find the indices of the dataset types in the parts
    dataset_indices = []
    for idx, part in enumerate(parts):
        for dataset_type in dataset_types:
            if dataset_type in part:
                dataset_indices.append((idx, dataset_type))
                break  # Stop checking other dataset types

    if len(dataset_indices) < 2:
        print(f"Skipping {filename}")
        return None

    # Model name is everything before the first dataset type
    first_idx, first_dataset_type = dataset_indices[0]
    model_name = '_'.join(parts[:first_idx + 1])

    train_dataset_type = first_dataset_type.capitalize()

    # Test dataset type is the second occurrence
    second_idx, second_dataset_type = dataset_indices[1]
    test_dataset_type = second_dataset_type.capitalize()

    # Number of variables in the test set is assumed to be the numeric part before the second-to-last "_"
    # Or between the dataset types

    num_vars = None
    # Try to find the number between first_idx and second_idx
    for idx in range(first_idx, second_idx):
        if parts[idx].isdigit():
            num_vars = int(parts[idx])
            break

    if num_vars is None:
        # Try to get the number before the second-to-last '_'
        if len(parts) >= 3 and parts[-3].isdigit():
            num_vars = int(parts[-3])
        else:
            print(f"Skipping {filename}, couldn't find number of variables.")
            return None

    return model_name, train_dataset_type, test_dataset_type, num_vars


def main():
    args = parse_arguments()
    directory = args.directory

    data = []

    # Iterate over files in the directory
    for filename in os.listdir(directory):
        if not filename.endswith('.txt'):
            continue

        parsed = parse_filename(filename)
        if parsed is None:
            continue

        model_name, train_dataset_type, test_dataset_type, num_vars = parsed

        # Now, read the file and compute accuracy
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r') as f:
            lines = f.readlines()

        total_samples = len(lines)
        correct_predictions = 0

        for idx, line in enumerate(lines):
            tokens = line.strip().split()
            if not tokens:
                continue

            last_token = tokens[-1]

            # Ground truth label: odd lines -> SAT, even lines -> UNSAT
            if (idx + 1) % 2 == 1:
                ground_truth = 'SAT'
            else:
                ground_truth = 'UNSAT'

            # Prediction
            if last_token in ['SAT', 'UNSAT']:
                prediction = last_token
            else:
                prediction = 'Wrong'

            if prediction == ground_truth:
                correct_predictions += 1

        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

        # Store the data
        data.append({
            'Model': model_name,
            'Train Dataset': train_dataset_type,
            'Test Dataset': test_dataset_type,
            'Num Vars': num_vars,
            'Accuracy': accuracy
        })

    # Create DataFrame
    df = pd.DataFrame(data)

    # Filter DataFrame to include only Test Dataset == 'Marginal'
    df_marginal = df[df['Test Dataset'] == 'Marginal']

    # Plotting
    plt.figure(figsize=(10, 6))

    # For each model, plot accuracy over variable numbers
    models = df_marginal['Model'].unique()
    for model in models:
        df_model = df_marginal[df_marginal['Model'] == model]
        df_model = df_model.sort_values('Num Vars')
        plt.plot(df_model['Num Vars'], df_model['Accuracy'], marker='o', label=model)

    # Add shaded region between x=6 and x=10
    plt.axvspan(6, 10, color='gray', alpha=0.2)

    # Get current axes and y-limits
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    ypos = (ymin + ymax) / 2  # Middle of y-axis

    # Add label for the shaded region
    plt.text(8, ypos, 'Training Regime', ha='center', va='center', color='gray', fontsize=12)

    plt.xlabel('Number of Variables')
    plt.ylabel('Accuracy')
    plt.title('Accuracy on Marginal Dataset')
    plt.legend()
    plt.grid(True, linestyle='--')  # Make grid lines dashed
    plt.show()


if __name__ == '__main__':
    main()
