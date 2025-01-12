# import os
# import pandas as pd
#
# # Define paths for valid, uncertain, detection, and label folders
# valid_folder = 'runs/detect/exp2/Valid_Predictions'
# uncertain_folder = 'runs/detect/exp2/Uncertainity_Predictions'
# detection_folder = 'data/images_25k'
# labels_folder = 'data/labels_25k'  # Folder where label files are stored
#
# # Get the list of image filenames in valid and uncertain folders
# valid_images = set(os.listdir(valid_folder))
# uncertain_images = set(os.listdir(uncertain_folder))
#
# # Prepare a list to store the names of deleted images
# deleted_images = []
#
# # Iterate through images in the valid folder
# for image in valid_images:
#     # Check if the image is also in the uncertain folder
#     if image not in uncertain_images:
#
#         image_path = os.path.join(detection_folder, image)
#         label_file = os.path.splitext(image)[0] + '.txt'  # Assuming labels have .txt extension
#         label_path = os.path.join(labels_folder, label_file)
#
#         # Delete the image from detection folder if it exists
#         if os.path.exists(image_path):
#             os.remove(image_path)
#             deleted_images.append(image)  # Add the image name to the deleted list
#             print(f"Removed image: {image}")
#
#         # Delete the corresponding label file if it exists
#         if os.path.exists(label_path):
#             os.remove(label_path)
#             print(f"Removed label file: {label_file}")
#
# # Create a DataFrame to save the deleted image names
# deleted_images_df = pd.DataFrame(deleted_images, columns=['Deleted Image Names'])
#
# # Save the DataFrame to an Excel file
# excel_file_path = 'deleted_images_after_second_detection.xlsx'
# deleted_images_df.to_excel(excel_file_path, index=False)
#
# print(f"Finished removing {len(deleted_images)} images and their corresponding label files.")
# print(f"Deleted image names saved to '{excel_file_path}'.")
#
import os
import pandas as pd

# Paths
csv_path = "runs/detect/dempster_second_epsilon_1/predictions.csv"  # Path to the CSV file containing predictions
image_folder = "C:/Thesis/images"  # Folder containing images

# Load the CSV file
df = pd.read_csv(csv_path)

# Assuming the CSV file has these columns: 'Image Name', 'Belief'
# Find images to delete
images_to_delete = []
deleted_images =[]
# Group predictions by image name
grouped = df.groupby('Image Name')
for image_name, group in grouped:
    # Check if all Belief scores for this image are >= 0.7
    if all(group['belief'] >= 0.7):
        images_to_delete.append(image_name)

# Delete images
for image_name in images_to_delete:
    image_path = os.path.join(image_folder, image_name)
    if os.path.exists(image_path):
        deleted_images.append(image_name)
        os.remove(image_path)
        print(f"Deleted: {image_path}")

print("Images with all Belief scores >= 0.7 have been deleted.")
import os
import pandas as pd
deleted_images_df = pd.DataFrame(deleted_images, columns=['Deleted Image Names'])
#
#Save the DataFrame to an Excel file
try:
    # Save as Excel if openpyxl is available
    excel_file_path = 'runs/detect/dempster_second_epsilon_1/deleted_images_after_second_detection.xlsx'
    deleted_images_df.to_excel(excel_file_path, index=False)
    print(f"Deleted image names saved to '{excel_file_path}'.")
except ModuleNotFoundError:
    # Fallback to saving as CSV
    csv_file_path = 'runs/detect/dempster_second_epsilon_1/deleted_images_after_second_detection.csv'
    deleted_images_df.to_csv(csv_file_path, index=False)
    print(f"openpyxl not installed. Deleted image names saved to '{csv_file_path}' as CSV.")



