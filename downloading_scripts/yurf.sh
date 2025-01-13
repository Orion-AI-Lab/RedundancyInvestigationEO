#!/bin/bash
if [ -z "$1" ]; then
  echo "No root path. Exiting!"
  exit 1
fi
root_folder_path=$1/"YURF_Shonai"

coastal_url=https://zenodo.org/record/3693326/files/Coastal_Forest_Data.zip?download=1
yurf_url=https://zenodo.org/record/3693326/files/Orthomosaics_TIFF%20file.zip?download=1

#Filenames for each url
filenames=("coastal.zip" "yurf.zip")
urls=($coastal_url $yurf_url)

#Create folder if it doesn't exist
yurf_dataset_folder="YURF"
shonai_dataset_folder="Coastal_Forest_Data"
mkdir $root_folder_path $root_folder_path/$yurf_dataset_folder $root_folder_path/$yurf_dataset_folder/"annotations"

#Download each url
full_path=($root_folder_path "$root_folder_path/$yurf_dataset_folder")
for ((index=0;index<${#filenames[@]}; index++))
do
    echo "Saving to ${full_path[index]}/"${filenames[index]}""
    # Use the wget command to download the file
    wget -O "${full_path[index]}"/"${filenames[index]}"  "${urls[index]}" 
    unzip "${full_path[index]}"/"${filenames[index]}" -d "${full_path[index]}"
    rm "${full_path[index]}"/"${filenames[index]}"
    echo "File "${filenames[index]}" downloaded successfully!"
done

folder_to_remove='Orthomosaics_TIFF file'
mv "$root_folder_path/$yurf_dataset_folder/$folder_to_remove"/* "$root_folder_path/$yurf_dataset_folder/"
rm -rf folder_to_remove

#Download YURF annotations

echo "Downloading YURF annotations"
wm1_url=https://zenodo.org/record/3693326/files/wM1.zip?download=1
wm2_url=https://zenodo.org/record/3693326/files/wM2.zip?download=1
wm3_url=https://zenodo.org/record/3693326/files/wM3.zip?download=1
wm4_url=https://zenodo.org/record/3693326/files/wM4.zip?download=1
wm5_url=https://zenodo.org/record/3693326/files/wM5.zip?download=1
wm6_url=https://zenodo.org/record/3693326/files/wM6.zip?download=1
wm7_url=https://zenodo.org/record/3693326/files/wM7.zip?download=1

wm_urls=($wm1_url $wm2_url $wm3_url $wm4_url $wm5_url $wm6_url $wm7_url)
wm_files=("wM1.zip" "wM2.zip" "wM3.zip" "wM4.zip" "wmM5.zip" "wM6.zip" "wM7.zip")
annotation_path=$root_folder_path/$yurf_dataset_folder/"annotations"

#Download each url
for ((index=0;index<${#wm_files[@]}; index++))
do
    echo "Saving to $annotation_path/"${wm_files[index]}""
    wget -O "$annotation_path"/"${wm_files[index]}"  "${wm_urls[index]}" 
    unzip "$annotation_path"/"${wm_files[index]}" -d "$annotation_path"
    rm "$annotation_path"/"${wm_files[index]}" 
    echo "File "${filenames[index]}" downloaded successfully!"
done

# Check if the download was successful
if [ $? -eq 0 ]; then
echo "Downloading finished normally."
else
echo "Downloading failed."
fi

echo "Process finished. Exiting!"