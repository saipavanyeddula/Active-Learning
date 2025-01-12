import fiftyone as fo
import fiftyone.zoo as foz

# The path to the source files that you manually downloaded
source_dir = "D:/bdd100k/images"
dataset = foz.load_zoo_dataset(
    "bdd10k",
    split="train",
    source_dir=source_dir,
    copy_files=False,
)

# The Dataset or DatasetView containing the samples you wish to export


# The directory to which to write the exported dataset
export_dir = "bdd10k_in_YOLOV5_train/"
# The type of dataset to export
# Any subclass of `fiftyone.types.Dataset` is supported
#Uncomment what ever format you wish to conver to
#YOLOV5
dataset_type = fo.types.YOLOv5Dataset  # for example
# Export the dataset
dataset.export(
    export_dir=export_dir,
    dataset_type=dataset_type
    #export_media="copy",
    #label_field=label_field,
)


# import os
#
# # Paths to your label files
# input_label_dir = r"D:\BDD100k\bdd_in_YOLOV5_train_newLabels\labels\val"
# output_label_dir = r"D:\BDD100k\bdd_in_YOLOV5_train_newLabels\labels_car_only"
#
# # Ensure output directory exists
# os.makedirs(output_label_dir, exist_ok=True)
#
# # Class ID for "car" (from your YAML file)
# car_class_id = "2"
#
# # Process each label file
# for label_file in os.listdir(input_label_dir):
#     if label_file.endswith(".txt"):
#         input_path = os.path.join(input_label_dir, label_file)
#         output_path = os.path.join(output_label_dir, label_file)
#
#         with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
#             for line in infile:
#                 # Split the line into components
#                 components = line.split()
#                 if components[0] == car_class_id:  # Check if class_id is "2"
#                     outfile.write(line)
#
# print(f"Filtered labels saved in {output_label_dir}")

