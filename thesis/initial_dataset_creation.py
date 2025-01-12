# import pandas as pd
#
# # Load the train CSV file
# train_csv = 'bdd100k_car_train_split_exactly_8000.csv'
# train_df = pd.read_csv(train_csv)
#
# # Create a stratify group based on weather, time of day, and scene
# train_df['stratify_group'] = train_df['weather'] + "_" + train_df['timeofday'] + "_" + train_df['scene']
#
# # Perform stratified sampling to select 5% of each group
# initial_train_df = train_df.groupby('stratify_group').apply(lambda x: x.sample(frac=0.05, random_state=42))
#
# # Reset index to avoid multi-index
# initial_train_df = initial_train_df.reset_index(drop=True)
#
# # Save the selected 5% images for initial training
# initial_train_df.to_csv('initial_train_5_percent_diversified.csv', index=False)
#
# print(f"Initial training set with 5% from each group: {len(initial_train_df)}")


# the below code is rounding up to extacly 400
import pandas as pd

# Load the train CSV file
train_csv = 'bdd100k_car_train_split_exactly_8000.csv'
train_df = pd.read_csv(train_csv)

# Create a stratify group based on weather, time of day, and scene
train_df['stratify_group'] = train_df['weather'] + "_" + train_df['timeofday'] + "_" + train_df['scene']

# Calculate the total samples to take (5% of 8000 is 400)
target_samples = 400

# Calculate the number of samples to select per group proportionally
group_sizes = train_df['stratify_group'].value_counts()
samples_per_group = (group_sizes / group_sizes.sum() * target_samples).round().astype(int)

# Ensure that the total number of samples adds up to target_samples
remaining_samples = target_samples - samples_per_group.sum()

# Adjust by adding remaining samples to the groups with the highest count
while remaining_samples > 0:
    for group in samples_per_group.sort_values(ascending=False).index:
        if remaining_samples == 0:
            break
        samples_per_group[group] += 1
        remaining_samples -= 1

# Perform stratified sampling based on the calculated number of samples per group
initial_train_df = train_df.groupby('stratify_group').apply(lambda x: x.sample(n=samples_per_group[x.name], random_state=42))

# Reset index to avoid multi-index
initial_train_df = initial_train_df.reset_index(drop=True)

# Save the selected 400 images for initial training
initial_train_df.to_csv('initial_train_400_images.csv', index=False)

print(f"Initial training set with 400 images: {len(initial_train_df)}")
