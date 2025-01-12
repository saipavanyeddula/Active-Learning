# import os
# import shutil
#
# import pandas as pd
#
# # Load the Excel file
# file_path = 'runs/detect/exp10/predictions_5.csv'
# df = pd.read_csv(file_path)
# print("Data preview:")
# print(df.head())
#
# # Sort by 'confs_score' in ascending order
# df_sorted = df.sort_values(by=df.columns[2], ascending=True)
# print(df_sorted.head())
#
# # Extract the first column (index 0, assumed to be the image names) and take only the top 200
# img_name_list = df_sorted.iloc[:200, 0].tolist()
#
# # Display the result
# print(img_name_list)
#
#
# # Define paths
# dataset_path = 'data/class_18_car'  # e.g., './dataset'
# images_folder = os.path.join(dataset_path, 'images')
# labels_folder = os.path.join(dataset_path, 'labels')
# train1_images_folder = './data/train5/images'  # e.g., './train1images'
# train1_labels_folder = './data/train5/labels'  # e.g., './train1labels'
#
# # Create destination folders if they don’t exist
# os.makedirs(train1_images_folder, exist_ok=True)
# os.makedirs(train1_labels_folder, exist_ok=True)
#
# # List of image names to extract (without extensions)
# img_name_list = [img_name.strip().replace('.jpg', '') for img_name in img_name_list]  # Remove any leading/trailing spaces and .jpg
# i = 1
# # Copy images and labels
# for img_name in img_name_list:
#     image_file = os.path.join(images_folder, f"{img_name}.jpg")  # or '.png' based on your file format
#     label_file = os.path.join(labels_folder, f"{img_name}.txt")
#
#     # Copy image if it exists
#     if os.path.exists(image_file):
#         shutil.copy(image_file, os.path.join(train1_images_folder, f"{img_name}.jpg"))
#
#     # Copy label if it exists
#     if os.path.exists(label_file):
#         shutil.copy(label_file, os.path.join(train1_labels_folder, f"{img_name}.txt"))
#         i+=1
#     print(i)
#
# print("Images and labels have been copied to the train1 folders.")

# import os
# import shutil
#
# # Paths
# image_folder = 'data/detect1/train/images'  # Folder containing the images
# label_folder = 'data/class_18_car/labels'  # Folder containing the labels
# output_folder = 'data/detect1/train/labels1'  # Folder to save the filtered labels
#
# # Create output folder if it doesn't exist
# os.makedirs(output_folder, exist_ok=True)
#
# # Get the list of image names without extensions
# image_names = {os.path.splitext(img)[0] for img in os.listdir(image_folder)}
#
# # Filter and copy corresponding labels
# for label_file in os.listdir(label_folder):
#     label_name = os.path.splitext(label_file)[0]
#     if label_name in image_names:  # Check if the label has a corresponding image
#         # Copy label to the output folder
#         src_path = os.path.join(label_folder, label_file)
#         dest_path = os.path.join(output_folder, label_file)
#         shutil.copy(src_path, dest_path)
#
# print("Filtered labels have been copied to the output folder.")

import os

# Paths to the folders
folder_A = "train2_dempster_epsilon2/images"  # Folder containing reference images
folder_B = "C:/Thesis/images"  # Folder to check and delete matching images

# Iterate through folder A
for file in os.listdir(folder_A):
    file_name, file_ext = os.path.splitext(file)

    # Construct the corresponding file path in folder B
    matching_file_path = os.path.join(folder_B, file_name + file_ext)

    # Check if the file exists in folder B
    if os.path.exists(matching_file_path):
        os.remove(matching_file_path)  # Delete the file
        print(f"Deleted: {matching_file_path}")

print("Matching images have been deleted from folder B.")
