#!/bin/bash
if [ -z "$1" ]; then
  echo "No root path. Exiting!"
  exit 1
fi

root_folder_path=$1
full_path="$root_folder_path/BigEarthNet"

mkdir -p $full_path

# Updated URLs for v2.0
# Base Zenodo record for v2.0: https://zenodo.org/records/10891137
sentinel2_url="https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst?download=1"
sentinel1_url="https://zenodo.org/records/10891137/files/BigEarthNet-S1.tar.zst?download=1"
ref_maps_url="https://zenodo.org/records/10891137/files/Reference_Maps.tar.zst?download=1"
metadata_url="https://zenodo.org/records/10891137/files/metadata.parquet?download=1"
snow_cloud_metadata_url="https://zenodo.org/records/10891137/files/metadata_for_patches_with_snow_cloud_or_shadow.parquet?download=1"

echo "Downloading Sentinel-2 data"
wget -O "$full_path/BigEarthNet-S2.tar.zst" $sentinel2_url

echo "Downloading Sentinel-1 data"
wget -O "$full_path/BigEarthNet-S1.tar.zst" $sentinel1_url

echo "Downloading Reference Maps"
wget -O "$full_path/Reference_Maps.tar.zst" $ref_maps_url

echo "Downloading Metadata Parquet files"
wget -O "$full_path/metadata.parquet" $metadata_url
wget -O "$full_path/metadata_for_patches_with_snow_cloud_or_shadow.parquet" $snow_cloud_metadata_url

echo "Untaring data..."
# Use tar with --zstd for .tar.zst files
for f in "$full_path"/*.tar.zst; do
    echo "Extracting $f"
    tar --zstd -xf "$f" --directory "$full_path"
    if [ $? -eq 0 ]; then
        echo "Extracted $(basename "$f") successfully."
        rm "$f"
    else
        echo "Extraction of $(basename "$f") failed."
    fi
done

echo "Download and setup for BigEarthNet finished."