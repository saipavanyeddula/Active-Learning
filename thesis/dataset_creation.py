# import os
# import shutil
# import pandas as pd
#
# # Define the paths for the source and destination folders
# source_image_folder = 'C:/Thesis/bdd_in_YOLOV5_train/images/val'  # e.g., 'bdd100k/images/train'
# source_label_folder = "C:/Thesis/bdd_in_YOLOV5_train/labels/cars_only_reassigned"  # e.g., 'bdd100k/labels/train'
#
# train_image_folder = "C:/Thesis/diversified_dataset_10k/train1/images"  # e.g., 'train_images'
# train_label_folder = "C:/Thesis/diversified_dataset_10k/train1/labels"  # e.g., 'train_labels'
# val_image_folder = "C:/Thesis/diversified_dataset_10k/valid/images"  # e.g., 'val_images'
# val_label_folder = "C:/Thesis/diversified_dataset_10k/valid/labels"  # e.g., 'val_labels'
#
# # Load the train and validation CSV files
# train_csv = 'initial_train_400_images.csv'
# val_csv = 'bdd100k_car_val_split_exactly_2000.csv'
#
# train_df = pd.read_csv(train_csv)
# val_df = pd.read_csv(val_csv)
#
# # Create target directories if they don't exist
# os.makedirs(train_image_folder, exist_ok=True)
# os.makedirs(train_label_folder, exist_ok=True)
# os.makedirs(val_image_folder, exist_ok=True)
# os.makedirs(val_label_folder, exist_ok=True)
#
#
# # Function to copy images and labels
# def copy_images_and_labels(df, image_folder, label_folder):
#     for _, row in df.iterrows():
#         image_id = row['image_id'].strip()  # Strip any leading/trailing spaces
#
#         # Check if the image_id already ends with '.jpg'
#         if not image_id.endswith('.jpg'):
#             image_id = image_id + '.jpg'  # Add .jpg if it doesn't exist
#
#         # Define source and target paths for image and label
#         source_image_path = os.path.join(source_image_folder, image_id)
#         source_label_path = os.path.join(source_label_folder,
#                                          image_id.replace('.jpg', '.txt'))  # Assuming label is .json
#
#         target_image_path = os.path.join(image_folder, image_id)
#         target_label_path = os.path.join(label_folder, image_id.replace('.jpg', '.txt'))  # Assuming label is .json
#
#         # Debugging print statements
#         print(f"Checking image: {source_image_path}")
#         print(f"Checking label: {source_label_path}")
#
#         # Copy image and label to the target directories
#         if os.path.exists(source_image_path) and os.path.exists(source_label_path):
#             shutil.copy(source_image_path, target_image_path)
#             shutil.copy(source_label_path, target_label_path)
#         else:
#             print(f"Warning: Missing file(s) for image {image_id}")
#
#
# # Copy for train dataset
# copy_images_and_labels(train_df, train_image_folder, train_label_folder)
#
# # Copy for validation dataset
# #copy_images_and_labels(val_df, val_image_folder, val_label_folder)
#
# print("Image and label copying completed.")

import os
import shutil
import pandas as pd

# Paths
csv_file = 'diverse_validation_images.csv'  # Update with your CSV file path
source_folder = "C:/Users/avari/Documents/Downloads/images"  # Update with your source folder path
destination_folder = 'detect/images'  # Update with your destination folder path

# Read CSV file
df = pd.read_csv(csv_file)


# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Iterate through image IDs and copy files
for image_id in df['image_id']:
    # Ensure the image_id ends with .jpg
    if not image_id.endswith('.jpg'):
        image_id += '.jpg'

    source_image = os.path.join(source_folder, image_id)
    destination_image = os.path.join(destination_folder, image_id)

    if os.path.exists(source_image):
        shutil.copy(source_image, destination_image)
        print(f"Copied: {source_image} to {destination_image}")
    else:
        print(f"Image not found: {source_image}")

print("Image copying completed.")


