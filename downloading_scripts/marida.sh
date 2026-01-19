#!/bin/bash

# 1. Validate Input Arguments
if [ -z "$1" ]; then
  echo "Error: No root path provided."
  echo "Usage: ./download_marida.sh <root_folder_path>"
  exit 1
fi

root_folder_path=$1
dataset_dir="$root_folder_path/MARIDA"

# 2. Setup Directory
if [ ! -d "$dataset_dir" ]; then
    echo "Creating directory: $dataset_dir"
    mkdir -p "$dataset_dir"
else
    echo "Directory $dataset_dir already exists."
fi

# 3. Define Download URL
# MARIDA v1.0 Zenodo Record: https://zenodo.org/records/5151941
# File: MARIDA.zip (~1.2 GB)
marida_url="https://zenodo.org/records/5151941/files/MARIDA.zip?download=1"
archive_path="$dataset_dir/MARIDA.zip"

# 4. Download Data
if [ -f "$archive_path" ]; then
    echo "Archive found at $archive_path. Skipping download."
else
    echo "Downloading MARIDA dataset from Zenodo..."
    wget -O "$archive_path" "$marida_url"

    # Check if download was successful
    if [ $? -eq 0 ]; then
        echo "Download finished successfully."
    else
        echo "Download failed. Please check your internet connection."
        # Remove partial file
        rm "$archive_path"
        exit 1
    fi
fi

# 5. Extract Data
echo "Extracting MARIDA data..."

# Check for unzip tool
if ! command -v unzip &> /dev/null; then
    echo "Error: 'unzip' command not found. Please install it (e.g., sudo apt install unzip)."
    exit 1
fi

# Unzip directly into the MARIDA folder
unzip -q "$archive_path" -d "$dataset_dir"

if [ $? -eq 0 ]; then
    echo "Extraction successful."
else
    echo "Extraction failed."
    exit 1
fi

# 6. Cleanup and Final Check
echo "Removing zip file..."
rm "$archive_path"

echo "----------------------------------------------------------------"
echo "MARIDA Setup Finished."
if [ -d "$dataset_dir/patches" ]; then
    echo "Dataset ready at: $dataset_dir"
    echo "Structure:"
    echo "  - Patches: $dataset_dir/patches"
    echo "  - Splits:  $dataset_dir/splits"
    echo "  - Labels:  $dataset_dir/mapping_labels.txt"
else
    echo "Warning: Expected 'patches' folder not found. Check extraction."
fi
echo "----------------------------------------------------------------"