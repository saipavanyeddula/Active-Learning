import pandas as pd
import json
from sklearn.model_selection import train_test_split

# Load the JSON file
json_path = 'archive (7)/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json'
with open(json_path, 'r') as f:
    data = json.load(f)

# Extract relevant data
records = []
for item in data:
    image_id = item['name']
    weather = item['attributes']['weather']
    timeofday = item['attributes']['timeofday']
    scene = item['attributes']['scene']

    # Simplify scene categories
    simplified_scene = scene if scene in ['city street', 'highway', 'residential'] else 'other'

    # Check if "car" exists in labels
    car_present = any(obj['category'] == 'car' for obj in item['labels'])
    if car_present:
        records.append([image_id, weather, timeofday, simplified_scene])

# Create a DataFrame
df = pd.DataFrame(records, columns=['image_id', 'weather', 'timeofday', 'scene'])

# Create stratify group by combining key attributes
df['stratify_group'] = df['weather'] + "_" + df['timeofday'] + "_" + df['scene']

# Count the occurrences of each group
group_counts = df['stratify_group'].value_counts()

# Filter out groups with less than 2 samples
valid_groups = group_counts[group_counts > 1].index
df_filtered = df[df['stratify_group'].isin(valid_groups)]

# Ensure we have exactly 10,000 samples
# Select 10,000 images with good diversity
selected_df = df_filtered.sample(n=10000, random_state=42)

# Check for any groups with only 1 sample after filtering
single_sample_groups = selected_df['stratify_group'].value_counts()[selected_df['stratify_group'].value_counts() == 1]

# If there are single sample groups, handle them separately
if len(single_sample_groups) > 0:
    print(f"Warning: The following groups have only one sample and will be excluded from stratification:")
    print(single_sample_groups)

    # Exclude single-sample groups from stratification
    selected_df_no_single_sample = selected_df[~selected_df['stratify_group'].isin(single_sample_groups.index)]

    # Perform stratified split on the remaining data with at least 2 samples
    train, val = train_test_split(
        selected_df_no_single_sample,
        test_size=0.2,  # 20% for validation
        stratify=selected_df_no_single_sample['stratify_group'],
        random_state=42
    )

    # Manually assign single-sample groups to either train or val
    # Alternatively, you could choose to assign single samples to whichever set you'd like (train or val)
    single_sample_train = single_sample_groups.index.to_list()
    single_sample_val = single_sample_train  # Or assign them to val as well

    # Append single-sample groups to train or validation
    train = pd.concat([train, selected_df[selected_df['stratify_group'].isin(single_sample_train)]])
    val = pd.concat([val, selected_df[selected_df['stratify_group'].isin(single_sample_val)]])

else:
    # Perform stratified split
    train, val = train_test_split(
        selected_df,
        test_size=0.2,  # 20% for validation
        stratify=selected_df['stratify_group'],
        random_state=42
    )

# Ensure the final total number of samples is exactly 10,000
# Adjusting the train and validation sets if needed
train = train.head(8000)
val = val.head(2000)

# Save the splits
train.to_csv("bdd100k_car_train_split_exactly_8000.csv", index=False)
val.to_csv("bdd100k_car_val_split_exactly_2000.csv", index=False)

print(f"Train set size: {len(train)}")
print(f"Validation set size: {len(val)}")
