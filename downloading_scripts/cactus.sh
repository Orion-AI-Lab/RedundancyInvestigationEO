#!/bin/bash
if [ -z "$1" ]; then
  echo "No root path. Exiting!"
  exit 1
fi
root_folder_path=$1

training_set_url="https://storage.googleapis.com/kaggle-data-sets/124738/300689/upload/training_set.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230508%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230508T153937Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=6dbfdec76c7139c31e9f22dc8c7c764d8186182c91cf1577fc36e0b84704a1e919b16a0d2d9c1956ce56d50fdb0d2e04cd0ec7b0294822ae2cfc5365fbc109f179a708b0178a0a215f998016420347dad51b723670beff84d5a82aa254feec8cfcbafad8ec436f8dbb2646d500e4224f0956c2e78e42ac1de3338a3619df2673be226d889ab68b7764b802cec767533f05294ce00982e824fe3863e5de3f86b70636386344698d04a4334d9bbb8192af31d257549f3e80c9af8f522f31c665c3ff00b0c652df27b850ceb1eb77ec135985e5ed6e4f654e069b02201e8181b32995e682b806f7ab2ade766fd33f7c694001baf4a4cdaccb27a624073005777e98"
validation_set_url="https://storage.googleapis.com/kaggle-data-sets/124738/300689/upload/validation_set.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230508%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230508T154007Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=29cf3b38f32441d0f379d8ee73768f7e5a825bdfc0ca9e898341df5e2556ab28eb8f24f99e9ad1d80cca18c9950e3e695947217cfdfc09bb1bf4c5501fdfb4e3f20a80e8baf63c67fe90ff1aaa899058f92a8dbb1051b9ccb1df00a51618b6abc38632640680cb397d8d779d7981457b36128b611dcef3b39a401a71b14273525078e14df0e27a26127405a76b5fbf67fd51f22aa6beaea60c50fe8b5dedbf6184d5c4d3af92a00c59155314e83cc00ad8b1833928cfd5d8d141748f42fa5fdc1a3f6e395a7c3dbf67e834ffb8d4bde1dbbd07934ec25f71879f01f1044d87e2b7e48438f9a3247a2d188b1a34e97f749b2bdb9b66fb90c71661845d479dcae5"

#Filenames for each url
filenames=(training_set.zip validation_set.zip)
urls=($training_set_url $validation_set_url)

#Create folder if it doesn't exist
dataset_folder="Cactus"
mkdir $root_folder_path/$dataset_folder

#Download each url
full_path="$root_folder_path/$dataset_folder"
for ((index=0;index<${#filenames[@]}; index++))
do
    echo "Saving to $full_path/"${filenames[index]}""
    # Use the curl command to download the file
    wget -O "$full_path"/"${filenames[index]}"  "${urls[index]}" 
    echo "File "${filenames[index]}" downloaded successfully!"
done

# Check if the download was successful
if [ $? -eq 0 ]; then
echo "Downloading finished normally."
else
echo "Downloading failed."
fi

echo "Extracting training data"

unzip "$full_path"/"training_set.zip" -d "$full_path"

echo "Extracting validation data"

unzip "$full_path"/"validation_set.zip" -d "$full_path"

echo "Cactus dataset has been saved successfully!"

echo "Removing zip files"
rm "$full_path"/"training_set.zip"
rm "$full_path"/"validation_set.zip"

echo "Process finished. Exiting!"