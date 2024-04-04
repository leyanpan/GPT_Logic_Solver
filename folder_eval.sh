#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 MODEL_PATH FOLDER"
    exit 1
fi

MODEL_PATH=$1
FOLDER=$2

# Iterate over each .txt file in the folder
for FILE_PATH in "$FOLDER"/*.txt; do
    # Check if the file is a regular file
    if [ -f "$FILE_PATH" ]; then
        FILE_NAME=$(basename "$FILE_PATH")
        echo "Evaluating $FILE_NAME"
        # Run the evaluation command for the file
        python eval.py "$MODEL_PATH" "$FOLDER" -f "$FILE_NAME"
    fi
done
