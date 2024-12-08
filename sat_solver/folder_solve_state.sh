#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

# Assign arguments to variables
input_dir=$1
output_dir=$2

# Check if input directory exists
if [ ! -d "$input_dir" ]; then
    echo "Error: Input directory '$input_dir' does not exist."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Loop through all txt files in the input directory
for input_file in "$input_dir"/*.txt; do
    # Get the base name of the file (without directory path)
    base_name=$(basename "$input_file")

    # Construct output file path
    output_file="$output_dir/${base_name%.txt}_State.txt"

    # Run the command for the input file
    python solve_trace.py "$input_file" "$output_file" -t state
done