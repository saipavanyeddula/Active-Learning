import pandas as pd
import shutil
import os

# Define paths
image_source_folder = "C:/Thesis/images"
label_source_folder = "D:/bdd_in_YOLOV5_train/cars_only_reassigned"
image_destination_folder = 'train2_dempster_epsilon2/images'
label_destination_folder = 'train2_dempster_epsilon2/labels'

# Load CSV file
df = pd.read_csv("D:/Active_learning/runs/detect/dempster_second_epsilon_1/selected_samples_with_normalized_ignorance_plus_doubt.csv")

# Iterate through each image name in the "Image Name" column
for index, row in df.iterrows():
    image_name = row['Image Name'].strip()  # Accessing by column name and stripping whitespace
    label_name = image_name.replace('.jpg', '.txt')  # Assuming labels have a .txt extension

    # Copy image
    source_image_path = os.path.join(image_source_folder, image_name)
    destination_image_path = os.path.join(image_destination_folder, image_name)

    print(f"Checking for image: {source_image_path}")  # Debugging output

    if os.path.exists(source_image_path):
        if not os.path.exists(destination_image_path):  # Check if already exists
            shutil.copy(source_image_path, destination_image_path)
            print(f"Copied image: {image_name}")
        else:
            print(f"Image {image_name} already exists in destination.")
    else:
        print(f"Image {image_name} not found at {source_image_path}.")

    # Copy label
    source_label_path = os.path.join(label_source_folder, label_name)
    destination_label_path = os.path.join(label_destination_folder, label_name)

    print(f"Checking for label: {source_label_path}")  # Debugging output

    if os.path.exists(source_label_path):
        if not os.path.exists(destination_label_path):  # Check if already exists
            shutil.copy(source_label_path, destination_label_path)
            print(f"Copied label: {label_name}")
        else:
            print(f"Label {label_name} already exists in destination.")
    else:
        print(f"Label {label_name} not found at {source_label_path}.")
