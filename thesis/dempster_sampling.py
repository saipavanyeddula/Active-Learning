import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load predictions CSV
predictions_csv = 'runs/detect/dempster_second_epsilon_1/predictions.csv'
data = pd.read_csv(predictions_csv)

# Ensure CSV has the correct headers
required_columns = ['Image Name', 'Class Name', 'Confidence Score', 'x_min', 'y_min', 'x_max', 'y_max', 'ignorance_plus_doubt']
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"CSV file must contain the following columns: {required_columns}")

# Normalize relevant columns including ignorance_plus_doubt
scaler = MinMaxScaler()
data[['Confidence Score', 'x_min', 'y_min', 'x_max', 'y_max', 'ignorance_plus_doubt']] = scaler.fit_transform(
    data[['Confidence Score', 'x_min', 'y_min', 'x_max', 'y_max', 'ignorance_plus_doubt']]
)

# Combine normalized values for clustering
clustering_features = data[['ignorance_plus_doubt', 'x_min', 'y_min', 'x_max', 'y_max']]

# Step 1: Compute Silhouette Scores for a range of cluster numbers
range_n_clusters = range(2, 10)
silhouette_scores = []
print(f"Finding optimal clusters...")
for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(clustering_features)
    score = silhouette_score(clustering_features, cluster_labels)
    silhouette_scores.append(score)

# Step 2: Find the optimal number of clusters
optimal_clusters = np.argmax(silhouette_scores) + range_n_clusters.start
print(f"Applying K-Means with optimal clusters: {optimal_clusters}")

# Step 3: Apply K-Means clustering
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(clustering_features)

# Step 4: Proportional sampling from clusters
total_samples = 500  # Total desired samples
min_threshold = 0.3  # Minimum normalized ignorance_plus_doubt threshold
max_threshold = 0.7  # Maximum normalized ignorance_plus_doubt threshold

# Filter data based on normalized ignorance_plus_doubt thresholds
filtered_data = data[
    (data['ignorance_plus_doubt'] >= min_threshold) &
    (data['ignorance_plus_doubt'] <= max_threshold)
]

# Calculate proportional samples for each cluster
cluster_counts = filtered_data['Cluster'].value_counts()
cluster_proportions = cluster_counts / cluster_counts.sum()
samples_per_cluster = (cluster_proportions * total_samples).astype(int)

# Ensure at least one sample per cluster, if possible
samples_per_cluster[samples_per_cluster == 0] = 1

selected_samples = pd.DataFrame(columns=filtered_data.columns)

for cluster_id, num_samples in samples_per_cluster.items():
    cluster_data = filtered_data[filtered_data['Cluster'] == cluster_id]
    sorted_cluster = cluster_data.sort_values('ignorance_plus_doubt', ascending=False)

    # Select the required number of samples or all available samples if fewer
    selected_samples_cluster = sorted_cluster.head(num_samples)
    selected_samples = pd.concat([selected_samples, selected_samples_cluster])

# Save selected samples to CSV
selected_samples_csv_path = 'runs/detect/dempster_second_epsilon_1/selected_samples_with_normalized_ignorance_plus_doubt.csv'
selected_samples[['Image Name', 'Class Name', 'ignorance_plus_doubt', 'x_min', 'y_min', 'x_max', 'y_max', 'Cluster']].to_csv(selected_samples_csv_path, index=False)

print(f"Selected {len(selected_samples)} samples for annotation saved to '{selected_samples_csv_path}'")

# Step 5: Plot Silhouette Scores and save the plot
silhouette_plot_path = 'runs/detect/dempster_second_epsilon_1/silhouette_scores_normalized.png'
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, silhouette_scores, marker='o', linestyle='--', color='b')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Optimal Number of Clusters Using Silhouette Score')
plt.grid()
plt.savefig(silhouette_plot_path)
plt.close()
print(f"Silhouette score plot saved to '{silhouette_plot_path}'")

# Step 6: Scatter plot for normalized ignorance_plus_doubt vs. clusters and save the plot
scatter_plot_path = 'runs/detect/dempster_second_epsilon_1/ignorance_plus_doubt_vs_clusters.png'
plt.figure(figsize=(10, 6))
for cluster_id in data['Cluster'].unique():
    cluster_data = data[data['Cluster'] == cluster_id]
    plt.scatter(cluster_data['ignorance_plus_doubt'], [cluster_id] * len(cluster_data),
                label=f'Cluster {cluster_id}', alpha=0.6)
plt.xlabel('Normalized Ignorance + Doubt')
plt.ylabel('Cluster ID')
plt.title('Normalized Ignorance + Doubt vs. Clusters')
plt.legend()
plt.grid()
plt.savefig(scatter_plot_path)
plt.close()
print(f"Scatter plot saved to '{scatter_plot_path}'")

# Step 7: Create 3D plot for Ignorance_plus_Doubt, bbox area, and clusters
data['Box Area'] = (data['x_max'] - data['x_min']) * (data['y_max'] - data['y_min'])
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(data['ignorance_plus_doubt'], data['Box Area'], data['Cluster'],
                     c=data['Cluster'], cmap='viridis', alpha=0.6)
ax.set_xlabel('Normalized Ignorance_plus_Doubt')
ax.set_ylabel('Bounding Box Area')
ax.set_zlabel('Cluster ID')
ax.set_title('3D Plot: Ignorance_plus_Doubt, Bounding Box Area, and Cluster ID')

fig.colorbar(scatter, ax=ax, label='Cluster ID')

plt.savefig('runs/detect/dempster_second_epsilon_1/ignorance_box_area_cluster_3d.png')
plt.close()
print("3D plot saved.")
