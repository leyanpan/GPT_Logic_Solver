import os
import sys
import subprocess
from tqdm import tqdm

# Check if the correct number of arguments are provided
if len(sys.argv) != 4:
    print("Usage: python script.py MODEL_PATH FOLDER OUTPUT_FILE")
    sys.exit(1)

MODEL_PATH = sys.argv[1]
FOLDER = sys.argv[2]
OUTPUT_FILE = sys.argv[3]

# Get the directory where the script is located (the root directory)
root_dir = os.path.dirname(os.path.abspath(__file__))

# Ensure the preds folder exists in the root directory (where this script is)
preds_folder = os.path.join(root_dir, 'preds')
if not os.path.exists(preds_folder):
    os.makedirs(preds_folder)
    print(f"Created 'preds' folder at {preds_folder}")

# Open the specified output file to log the results
with open(OUTPUT_FILE, "w") as log_file:

    # List all .txt files in the folder
    txt_files = [
        f for f in os.listdir(FOLDER)
        if f.endswith('.txt') and os.path.isfile(os.path.join(FOLDER, f))
    ]

    # Initialize the tqdm progress bar
    with tqdm(total=len(txt_files), desc="Processing Files", unit="file") as pbar:

        # Iterate over each .txt file in the folder
        for file_name in txt_files:
            file_path = os.path.join(FOLDER, file_name)

            log_file.write(f"Evaluating {file_name}\n")

            # Run the evaluation command for the file, redirecting stderr to stdout
            process = subprocess.Popen(
                ['python', '-u', 'eval.py', MODEL_PATH, FOLDER, '-f', file_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Redirect stderr to stdout
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Read output in real-time
            for stdout_line in iter(process.stdout.readline, ""):
                log_file.write(stdout_line)  # Write to log file
                sys.stdout.write(stdout_line)  # Print to terminal
                sys.stdout.flush()  # Ensure real-time output

            # Wait for the process to complete
            process.stdout.close()
            process.wait()

            # Update the progress bar
            pbar.update(1)

    log_file.write("Evaluation complete.\n")
