#!/bin/bash
if [ -z "$1" ]; then
  echo "No root path. Exiting!"
  exit 1
fi
root_folder_path=$1

dataset_folder="GLAD"

full_path="$root_folder_path/$dataset_folder"

mkdir $full_path "$full_path/datamask" "$full_path/treecover" "$full_path/gain" "$full_path/lossyear" "$full_path/first" "$full_path/last"

coordinates=("00N_020E" "10N_020E" "00N_010E" "10S_010E" "10S_020E" "10N_080W" "10N_070W" "10N_060W" "00N_080W" "00N_070W" "00N_060W" "00N_050W" "00N_040W" "10S_080W" "10S_070W" "10S_060W" "10S_050W" "20S_070W" "20S_060W" "20S_050W" "30S_070W" "30S_060W" "40S_080W" "40S_070W" "50S_080W" "50S_070W" "20N_090W" "20N_100W" "30N_110W" "70N_100E" "70N_110E" "70N_120E" "70N_130E" "70N_140E" "70N_150E" "70N_160E" "70N_170E" "60N_100E" "60N_110E" "60N_120E" "60N_130E" "20N_020W" "20N_010W" "20N_000E" "20N_010E" "20N_020E" "20N_030E" "20N_040E" "10N_090E" "10N_100E" "00N_100E" "00N_110E" "30N_090E" "30N_100E" "20N_090E" "20N_100E")
datamask_url="https://storage.googleapis.com/earthenginepartners-hansen/GFC-2022-v1.10/Hansen_GFC-2022-v1.10_datamask_"
treecover_url="https://storage.googleapis.com/earthenginepartners-hansen/GFC-2022-v1.10/Hansen_GFC-2022-v1.10_treecover2000_"
gain_url="https://storage.googleapis.com/earthenginepartners-hansen/GFC-2022-v1.10/Hansen_GFC-2022-v1.10_gain_"
lossyear_url="https://storage.googleapis.com/earthenginepartners-hansen/GFC-2022-v1.10/Hansen_GFC-2022-v1.10_lossyear_"
first_url="https://storage.googleapis.com/earthenginepartners-hansen/GFC-2022-v1.10/Hansen_GFC-2022-v1.10_first_"
last_url="https://storage.googleapis.com/earthenginepartners-hansen/GFC-2022-v1.10/Hansen_GFC-2022-v1.10_last_"

for coord in "${coordinates[@]}"
do 
wget -P "$full_path/datamask/" "$datamask_url$coord.tif"
wget -P "$full_path/treecover/" "$treecover_url$coord.tif"
wget -P "$full_path/gain/" "$gain_url$coord.tif"
wget -P "$full_path/lossyear/" "$lossyear_url$coord.tif"
wget -P "$full_path/first/" "$first_url$coord.tif"
wget -P "$full_path/last/" "$last_url$coord.tif"
echo $coord " was downloaded successfully!"
done