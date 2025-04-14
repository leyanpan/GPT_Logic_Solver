import os
import sys
import re
import importlib
from pathlib import Path

# Example function (can be in the same file or imported from another module)
def remove_up(s: str) -> str:
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

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <folder> <function_name>")
        sys.exit(1)

    folder = sys.argv[1]
    function_name = sys.argv[2]

    if not os.path.isdir(folder):
        print(f"Error: {folder} is not a valid directory.")
        sys.exit(1)

    # Get the function from the current module
    func = globals().get(function_name)
    if not callable(func):
        print(f"Error: Function '{function_name}' not found.")
        sys.exit(1)

    output_folder = f"{folder}_{function_name}"
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(folder):
        if file_name.endswith('.txt'):
            input_path = os.path.join(folder, file_name)
            output_path = os.path.join(output_folder, file_name)

            with open(input_path, 'r', encoding='utf-8') as infile, \
                 open(output_path, 'w', encoding='utf-8') as outfile:

                for line in infile:
                    transformed = func(line.strip())
                    outfile.write(transformed + '\n')

    print(f"Transformation complete. Files written to: {output_folder}")

if __name__ == "__main__":
    main()
