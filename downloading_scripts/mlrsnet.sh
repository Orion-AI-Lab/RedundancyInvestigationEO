#!/bin/bash

# Validate arguments
if [ -z "$1" ]; then
  echo "Error: No root path provided."
  echo "Usage: ./download_mlrsnet.sh <root_folder_path> <optional_download_url>"
  exit 1
fi

root_folder_path=$1
# Users can pass the direct URL as the second argument, or we use a placeholder
download_url=${2:-""}

full_path="$root_folder_path/MLRSNet"

# Create directory structure
if [ ! -d "$full_path" ]; then
    echo "Creating directory: $full_path"
    mkdir -p "$full_path"
else
    echo "Directory $full_path already exists."
fi

# 1. Handle the Download
archive_name="MLRSNet_V1.zip"
archive_path="$full_path/$archive_name"

if [ -f "$archive_path" ]; then
    echo "Archive already exists at $archive_path. Skipping download."
else
    if [ -z "$download_url" ]; then
        echo "----------------------------------------------------------------"
        echo "(!) ACTION REQUIRED: MLRSNet uses temporary download links."
        echo "1. Go to: https://data.mendeley.com/datasets/7j9bv9vwsx/1"
        echo "2. Click 'Download All' (or right-click and copy link address)."
        echo "3. Run this script again with the URL as the second argument:"
        echo "   ./download_mlrsnet.sh $root_folder_path \"<PASTE_LINK_HERE>\""
        echo "----------------------------------------------------------------"
        exit 1
    else
        echo "Downloading MLRSNet data..."
        # Mendeley links often require outputting to a specific filename
        wget -O "$archive_path" "$download_url"
        
        if [ $? -eq 0 ]; then
            echo "Downloading finished normally."
        else
            echo "Downloading failed. The link may have expired."
            rm "$archive_path" # Clean up partial file
            exit 1
        fi
    fi
fi

# 2. Extract Data
echo "Unzipping data (this may take a while)..."

# Check if unzip is installed
if ! command -v unzip &> /dev/null; then
    echo "Error: 'unzip' is not installed. Please install it (e.g., sudo apt install unzip)."
    exit 1
fi

# MLRSNet is usually a zip file containing 'Images' and 'Labels' folders
unzip -q "$archive_path" -d "$full_path"

if [ $? -eq 0 ]; then
    echo "Extracted MLRSNet data successfully."
else
    echo "Extraction failed."
    exit 1
fi

# 3. Cleanup
echo "Removing archive file..."
rm "$archive_path"

# 4. Verification (Optional)
if [ -d "$full_path/Images" ] && [ -d "$full_path/Labels" ]; then
    echo "----------------------------------------------------------------"
    echo "Download & Setup Finished."
    echo "Images: $full_path/Images"
    echo "Labels: $full_path/Labels"
    echo "----------------------------------------------------------------"
else
    echo "Warning: Standard 'Images' or 'Labels' folders not found."
    echo "Check the extraction folder: $full_path"
fi