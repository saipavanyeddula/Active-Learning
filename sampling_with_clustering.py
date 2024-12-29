import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load predictions CSV
predictions_csv = 'runs/detect/exp37/predictions.csv'  # Path to your predictions CSV
data = pd.read_csv(predictions_csv)

# Ensure CSV has the correct headers
required_columns = ['Image Name', 'Class Name', 'Confidence Score', 'x_min', 'y_min', 'x_max', 'y_max']
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"CSV file must contain the following columns: {required_columns}")

# Normalize confidence scores and bounding box coordinates
scaler = MinMaxScaler()
data[['Confidence Score', 'x_min', 'y_min', 'x_max', 'y_max']] = scaler.fit_transform(
    data[['Confidence Score', 'x_min', 'y_min', 'x_max', 'y_max']]
)

# Combine confidence and bounding box coordinates for clustering
clustering_features = data[['Confidence Score', 'x_min', 'y_min', 'x_max', 'y_max']]

# Apply k-means clustering (choose k based on your desired number of clusters)
kmeans = KMeans(n_clusters=5, random_state=42)  # Adjust n_clusters as needed
data['Cluster'] = kmeans.fit_predict(clustering_features)

# Add cluster labels to data
print("Clustered Predictions:")
print(data[['Image Name', 'Class Name', 'Confidence Score', 'x_min', 'y_min', 'x_max', 'y_max', 'Cluster']])

# Define the number of samples to select per cluster and confidence thresholds
samples_per_cluster = 10  # Adjust based on your annotation budget
min_confidence_threshold = 0.2  # Minimum confidence score for inclusion
max_confidence_threshold = 0.6  # Maximum confidence score for inclusion

selected_samples = []

# Iterate through clusters
for cluster_id in data['Cluster'].unique():
    cluster_data = data[data['Cluster'] == cluster_id]
    # Filter data based on confidence thresholds
    cluster_data = cluster_data[
        (cluster_data['Confidence Score'] >= min_confidence_threshold) &
        (cluster_data['Confidence Score'] <= max_confidence_threshold)
    ]
    # Sort by confidence score (least confident first)
    sorted_cluster = cluster_data.sort_values('Confidence Score', ascending=True)
    # Select up to N samples from this cluster
    selected_samples.append(sorted_cluster.head(samples_per_cluster))

# Combine selected samples
selected_samples_df = pd.concat(selected_samples)

# Save selected samples to a new CSV for annotation
selected_samples_csv_path = 'runs/detect/exp37/selected_samples_with_thresholds.csv'
selected_samples_df.to_csv(selected_samples_csv_path, index=False)
print(f"Selected samples for annotation saved to '{selected_samples_csv_path}'")

# Scatter plot for confidence score vs. bounding box clusters
plt.figure(figsize=(10, 6))
for cluster_id in data['Cluster'].unique():
    cluster_data = data[data['Cluster'] == cluster_id]
    plt.scatter(cluster_data['Confidence Score'], [cluster_id] * len(cluster_data),
                label=f'Cluster {cluster_id}', alpha=0.6)

plt.xlabel('Confidence Score')
plt.ylabel('Cluster ID')
plt.title('Confidence Score vs. Clusters')
plt.legend()
plt.grid()
plt.show()



# import pandas as pd
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import MinMaxScaler
#
# # Load predictions CSV
# predictions_csv = 'runs/detect/exp37/predictions.csv'  # Path to your predictions CSV
# data = pd.read_csv(predictions_csv)
#
# # Normalize confidence scores and bounding box coordinates
# scaler = MinMaxScaler()
# data[['Confidence Score', 'x_min', 'y_min', 'x_max', 'y_max']] = scaler.fit_transform(
#     data[['Confidence Score', 'x_min', 'y_min', 'x_max', 'y_max']]
# )
#
# # Combine confidence and bounding box coordinates for clustering
# clustering_features = data[['Confidence Score', 'x_min', 'y_min', 'x_max', 'y_max']]
#
# # Apply k-means clustering (choose k based on your desired number of clusters)
# kmeans = KMeans(n_clusters=5, random_state=42)  # Adjust n_clusters as needed
# data['Cluster'] = kmeans.fit_predict(clustering_features)
#
# # Add cluster labels to data
# print("Clustered Predictions:")
# print(data[['Image Name', 'Class Name', 'Confidence Score', 'Bounding Box', 'Cluster']])
#
#
# # Define the number of samples to select per cluster
# samples_per_cluster = 10  # Adjust based on your annotation budget
# selected_samples = []
#
# # Iterate through clusters
# for cluster_id in data['Cluster'].unique():
#     cluster_data = data[data['Cluster'] == cluster_id]
#     # Sort by confidence score (least confident first)
#     sorted_cluster = cluster_data.sort_values('Confidence Score', ascending=True)
#     # Select top N samples from this cluster
#     selected_samples.append(sorted_cluster.head(samples_per_cluster))
#
# # Combine selected samples
# selected_samples_df = pd.concat(selected_samples)
#
# # Save selected samples to a new CSV for annotation
# selected_samples_df.to_csv('selected_samples.csv', index=False)
# print("Selected samples for annotation saved to 'selected_samples.csv'")
#
#
# import matplotlib.pyplot as plt
#
# # Scatter plot for confidence score vs. bounding box clusters
# plt.figure(figsize=(10, 6))
# for cluster_id in data['Cluster'].unique():
#     cluster_data = data[data['Cluster'] == cluster_id]
#     plt.scatter(cluster_data['Confidence Score'], cluster_id, label=f'Cluster {cluster_id}', alpha=0.6)
#
# plt.xlabel('Confidence Score')
# plt.ylabel('Cluster ID')
# plt.title('Confidence Score vs. Clusters')
# plt.legend()
# plt.grid()
# plt.show()
