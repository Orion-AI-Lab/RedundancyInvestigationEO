#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys

CONFIG_FILE = "configs/download/download.json"

def load_json_with_comments(filepath):
    with open(filepath, "r") as f:
        content = f.read()
    content = re.sub(r'//.*', '', content)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    return json.loads(content)

def main():
    if not os.path.isfile(CONFIG_FILE):
        print(f"Error: Configuration file '{CONFIG_FILE}' not found.")
        sys.exit(1)

    try:
        config = load_json_with_comments(CONFIG_FILE)
    except Exception as e:
        print(f"Error: Failed to parse '{CONFIG_FILE}': {e}")
        sys.exit(1)

    dataset = config.get("dataset")
    root_path = config.get("root_path")

    if not dataset or not root_path:
        print("Error: 'dataset' and 'root_path' must be specified in download.json.")
        sys.exit(1)

    script_map = {
        "mlrsnet": "./downloading_scripts/mlrsnet.sh"
    }

    script_path = script_map.get(dataset.lower())

    if not script_path or not os.path.isfile(script_path):
        print(f"Error: Execution script for '{dataset}' not found at {script_path}")
        sys.exit(1)

    print(f"Starting download for dataset: {dataset}")
    print(f"Target destination: {root_path}")
    print(f"Executing script: {script_path}\n")

    # Pass ONLY the root_path to the shell script
    try:
        subprocess.run([script_path, root_path], check=True)
        print(f"\nSuccessfully finished downloading {dataset}!")
    except subprocess.CalledProcessError as e:
        print(f"\nError: Script execution failed with exit code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()