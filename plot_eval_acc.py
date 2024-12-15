import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

# Import the parse_compiled_eval_files function
from theory.plot_exactness import parse_compiled_eval_files

def parse_filename(filename):
    filename_lower = filename.lower().replace('.txt', '')

    dataset_types = ['marginal', 'random', 'skewed']

    parts = filename_lower.split('_')

    # Find the indices of the dataset types in the parts
    dataset_indices = []
    for idx, part in enumerate(parts):
        for dataset_type in dataset_types:
            if dataset_type == part:
                dataset_indices.append((idx, dataset_type))
                break  # Stop checking other dataset types

    if len(dataset_indices) < 2:
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
    parser = argparse.ArgumentParser(description="Process dataset files and compute accuracies.")
    parser.add_argument('directory', type=str, help="Directory containing dataset files")
    parser.add_argument('-l', '--train_var_min', type=int, default=6, help="Minimum number of variables in the training regime")
    parser.add_argument('-r', '--train_var_max', type=int, default=10, help="Maximum number of variables in the training regime")
    parser.add_argument('-b', '--beta', type=float, default=20, help="Mean Exactness beta value for compiled models")
    parser.add_argument('-c', '--compiled_dir', type=str, default=None, help="Directory containing compiled model evaluation results")
    parser.add_argument('-t', '--eval_type', type=str, choices=['Marginal', 'Random', 'Skewed'], default='Marginal', help="Evaluation dataset type to plot")
    args = parser.parse_args()
    directory = args.directory
    train_var_min = args.train_var_min
    train_var_max = args.train_var_max

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

    # Filter DataFrame to include variable numbers within training regime
    df_training_regime = df[(df['Num Vars'] >= train_var_min) & (df['Num Vars'] <= train_var_max)]

    # Compute the 9 accuracy results
    train_datasets = ['Marginal', 'Skewed', 'Random']
    test_datasets = ['Marginal', 'Skewed', 'Random']
    accuracy_results = []

    for train_dataset in train_datasets:
        for test_dataset in test_datasets:
            df_subset = df_training_regime[(df_training_regime['Train Dataset'] == train_dataset) &
                                           (df_training_regime['Test Dataset'] == test_dataset)]
            avg_accuracy = df_subset['Accuracy'].mean()
            accuracy_results.append({
                'Train Dataset': train_dataset,
                'Test Dataset': test_dataset,
                'Average Accuracy': avg_accuracy
            })

    # Output the accuracy results
    print("Average Accuracies within Training Regime (Variables {} to {}):".format(train_var_min, train_var_max))
    for result in accuracy_results:
        print("Train Dataset: {}, Test Dataset: {}, Average Accuracy: {:.2%}".format(
            result['Train Dataset'], result['Test Dataset'], result['Average Accuracy']))

    # Filter DataFrame to include only Test Dataset == args.eval_type
    df_eval = df[df['Test Dataset'] == args.eval_type]

    # Plotting
    plt.figure(figsize=(10, 6))

    # Assign colors based on training dataset
    color_map = {
        'Random': 'blue',
        'Marginal': 'orange',
        'Skewed': 'green',
        'Compiled': 'purple'
    }

    # For each model, plot accuracy over variable numbers
    models = df_eval['Model'].unique()
    for model in models:
        split_name = model.split('_')
        model_l, model_r = split_name[0], split_name[1]
        if int(model_l) != train_var_min:
            print(f"Skipping model {model} that does not match training regime.")
            continue
        df_model = df_eval[df_eval['Model'] == model]
        df_model = df_model.sort_values('Num Vars')
        train_dataset = df_model['Train Dataset'].iloc[0]  # Assuming all entries have the same train dataset
        color = color_map.get(train_dataset, 'black')  # Default to black if not found

        plt.plot(df_model['Num Vars'], df_model['Accuracy'] * 100, marker='o', label=model, color=color)

    # If compiled_dir is provided, load compiled model data
    if args.compiled_dir:
        df_compiled = parse_compiled_eval_files(args.compiled_dir)
        if df_compiled is not None:
            # Filter data for the specified beta and eval_type
            df_compiled_beta = df_compiled[(df_compiled['beta'] == args.beta) & (df_compiled['Test Dataset'] == args.eval_type)]
            if not df_compiled_beta.empty:
                # Sort by num_vars
                df_compiled_beta = df_compiled_beta.sort_values('num_vars')
                # Plot the data
                plt.plot(df_compiled_beta['num_vars'], df_compiled_beta['accuracy'], marker='o', label='Compiled', color=color_map['Compiled'])
            else:
                print(f"No compiled data found for beta={args.beta} and Test Dataset={args.eval_type}")

    # Add shaded region between x=train_var_min and x=train_var_max
    plt.axvspan(train_var_min, train_var_max, color='gray', alpha=0.2)

    # Get current axes and y-limits
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    ypos = (ymin + ymax) / 2  # Middle of y-axis

    # Add label for the shaded region
    plt.text((train_var_min + train_var_max) / 2, ypos, 'Training Regime',
             ha='center', va='center', color='gray', fontsize=12)

    plt.xlabel('Number of Variables')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy on {args.eval_type} Dataset')
    plt.legend()
    plt.grid(True, linestyle='--')  # Make grid lines dashed
    plt.show()

if __name__ == '__main__':
    main()