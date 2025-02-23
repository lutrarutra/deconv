#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <version>"
    exit 1
fi

SCRIPT_PATH="$(readlink -f "$0")"
SOURCE_FOLDER=$(dirname "$SCRIPT_PATH")
VERSION=$1
FOLDER_NAME="deconv.$VERSION"
ZIP_FILE="$FOLDER_NAME.zip"

# Check if the source folder exists
if [ ! -d "$SOURCE_FOLDER" ]; then
    echo "Error: Source folder '$SOURCE_FOLDER' does not exist."
    exit 1
fi

# Create the new version folder
mkdir "$FOLDER_NAME" || { echo "Failed to create directory $FOLDER_NAME"; exit 1; }

# Copy files from the source folder to the new version folder
cp -r "$SOURCE_FOLDER/DeconV" "$FOLDER_NAME/" || { echo "Failed to copy files from $SOURCE_FOLDER"; exit 1; }
cp -r "$SOURCE_FOLDER/README.md" "$FOLDER_NAME/" || { echo "Failed to copy files from $SOURCE_FOLDER"; exit 1; }
cp -r "$SOURCE_FOLDER/setup.py" "$FOLDER_NAME/" || { echo "Failed to copy files from $SOURCE_FOLDER"; exit 1; }
cp -r "$SOURCE_FOLDER/requirements.txt" "$FOLDER_NAME/" || { echo "Failed to copy files from $SOURCE_FOLDER"; exit 1; }

# Create a zip file of the new version folder
zip -r "$ZIP_FILE" "$FOLDER_NAME" || { echo "Failed to create zip file $ZIP_FILE"; exit 1; }

rm -r "$FOLDER_NAME"

echo "Created zip file: $ZIP_FILE"

echo "Building conda package..."

conda build purge
# --ouput gives the output path of the built package (i believe there is bug, that's why we need to build it twice)
build_path=$(conda build $SOURCE_FOLDER --output)
conda build $SOURCE_FOLDER --debug

# conda convert --platform all $build_path
echo "Built conda package: $build_path"
echo "Uploading to Anaconda.org..."

anaconda upload "$build_path"