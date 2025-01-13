#!/bin/bash
if [ -z "$1" ]; then
  echo "No root path. Exiting!"
  exit 1
fi
root_folder_path=$1

download_url=https://zenodo.org/record/4054338/files/IndividualSickFirDetectionDataset.zip?download=1

full_path="$root_folder_path/SickFir2"

mkdir $full_path

echo "Downloading Data"

wget -O "$full_path/sickfir.zip" $download_url

# Check if the download was successful
if [ $? -eq 0 ]; then
echo "Downloading finished normally."
else
echo "Downloading failed."
fi

unzip "$full_path/sickfir.zip" -d $full_path

echo "Removing zip files"

rm "$full_path/waititu.zip"