import os
import sys
import re

def remove_states_followed_by_up(s: str) -> str:
    """
    Remove all states followed by the [UP] token and the [UP] token themselves.
    """
    separators = r'\[SEP\]|\[BT\]|\[UP\]'
    parts = re.split(f'({separators})', s)
    output = []
    i = 0

    while i < len(parts):
        part = parts[i]
        if part == '[UP]':
            if output and re.match(separators, output[-1]) is None:
                output.pop()
        elif part.strip() != '[UP]':
            output.append(part)
        i += 1

    return ''.join(output).strip()

def process_file(file_path):
    """
    Process a single file: remove states followed by [UP], count splits after first [SEP].
    """
    count = 0

    with open(file_path, 'r') as f:
        for line in f:
            # Step 1: Apply the function to clean the line
            cleaned_line = remove_states_followed_by_up(line)

            # Step 2: Find the portion after the first [SEP]
            if '[SEP]' in cleaned_line:
                after_sep = cleaned_line.split('[SEP]', 1)[1]

                # Step 3: Split this portion using spaces and count the splits
                count = max(len(after_sep.split()), count)

    return count

def main(folder_path):
    """
    Process all txt files in the given folder.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        sys.exit(1)

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            count = process_file(file_path)
            print(f"{file_name}: {count}")

if __name__ == "__main__":
    # Ensure a folder path is provided as an argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    main(folder_path)