# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 12:24:03 2025

@author: fankoh-6
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

#%% Read file containing point cloud data
pcd = np.load("dataset2.npy")

pcd.shape
#%% Utility function for visualizing point cloud
def show_cloud(points_plt, labels=None, downsample=1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    if labels is None:
        ax.scatter(points_plt[::downsample, 0], points_plt[::downsample, 1], points_plt[::downsample, 2], s=0.5)
    else:
        ax.scatter(points_plt[::downsample, 0], points_plt[::downsample, 1], points_plt[::downsample, 2], c=labels[::downsample], s=0.5)
    
    ax.set_title("Point Cloud Visualization")
    plt.show()
    
#%% Show initial point cloud
show_cloud(pcd, downsample=10)  # Downsampled point cloud for quick visualization   
 
#%% Find best ground level

# Number of bins based on Sturges rule
n= int(1+3.3*np.log10(len(pcd[:, 2])))
print(n)

def get_ground_level(pcd, bins=n):
    hist, bin_edges = np.histogram(pcd[:, 2], bins=bins)
    max_bin_index = np.argmax(hist)  
    ground_level = bin_edges[max_bin_index + 1]
    return ground_level

# Estimate ground level
est_ground_level = get_ground_level(pcd)
print(f"Estimated ground level: {est_ground_level}")

#%% Plot histogram
n, bins, patches= plt.hist(pcd[:, 2], bins=n, alpha=0.75)
plt.axvline(est_ground_level, color='red', linestyle='dashed', linewidth=2)
plt.title("Ground Level Histogram")
plt.xlabel("Height (Z)")
plt.ylabel("Frequency")
print(bins) 
plt.show()

#%% Remove ground level
pcd_above_ground = pcd[pcd[:,2] > est_ground_level] 
pcd_above_ground.shape

#%% side view
show_cloud(pcd_above_ground, downsample=10)

#%% Task 2 - Find optimal eps for DBSCAN
def calculate_k_dist(pcd_above_ground, k):
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(pcd_above_ground)
    distances, _ = neighbors_fit.kneighbors(pcd_above_ground)
    k_dist = np.sort(distances[:, -1])  # Sort distances to the kth nearest neighbour
    return k_dist

def find_knee_point(k_dist_sorted):
    dy = np.diff(k_dist_sorted)
    ddy = np.diff(dy)

    # Take the average of the 10 largest changes in dervative
    top_indices = np.argsort(ddy)[-50:]  
    avg_index = int(np.mean(top_indices))  
    optimal_eps = k_dist_sorted[avg_index]

    # Plot the elbow
    plt.figure(figsize=(8, 6))
    plt.plot(k_dist_sorted, label="k-dist", color="darkgreen")
    plt.scatter(avg_index, optimal_eps, color="red", s=50, label="Optimal eps")
    plt.xlabel("Data Points Sorted")
    plt.ylabel("Distance to k-th Nearest Neighbor")
    plt.title("Elbow Method for Optimal Epsilon")
    plt.legend()
    plt.grid(True)
    plt.show()

    return optimal_eps

k = 5  
k_dist_sorted = calculate_k_dist(pcd_above_ground, k)
optimal_eps = find_knee_point(k_dist_sorted)
print(f"Optimal eps: {optimal_eps}")

# Apply DBSCAN
clustering = DBSCAN(eps=optimal_eps, min_samples=5).fit(pcd_above_ground)
labels = clustering.labels_
#%%
clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, clusters)]

# %% Plotting resulting clusters

plt.figure(figsize=(10,10))
plt.scatter(pcd_above_ground[:,0], 
            pcd_above_ground[:,1],
            c=clustering.labels_,
            cmap="tab20",
            s=2)


plt.title('DBSCAN: %d clusters' % clusters,fontsize=20)
plt.xlabel('x axis',fontsize=14)
plt.ylabel('y axis',fontsize=14)
plt.show()

#%% Task 3 - Find the largest cluster

unique_labels, counts = np.unique(labels, return_counts=True)

if -1 in unique_labels:
    noise_index = np.where(unique_labels == -1)
    unique_labels = np.delete(unique_labels, noise_index)
    counts = np.delete(counts, noise_index)

# Create variable to store information about the largest cluster
max_height = 0
largest_cluster_label = None
largest_cluster = None

for label in unique_labels:
    cluster_points = pcd_above_ground[labels == label]
    
    #Min & max for y-coordinates (height)
    min_y = np.min(cluster_points[:, 1])
    max_y = np.max(cluster_points[:, 1])
    height = max_y - min_y
    
    if height > max_height:
        max_height = height
        largest_cluster_label = label
        largest_cluster = cluster_points

if largest_cluster is not None:
    print(f"The largest cluster (likely catenary) has height: {max_height} with label: {largest_cluster_label}")
else:
    print("No clusters found.")

min_x = np.min(largest_cluster[:, 0])
max_x = np.max(largest_cluster[:, 0])
min_y = np.min(largest_cluster[:, 1])
max_y = np.max(largest_cluster[:, 1])

print(f"Cluster bounds: min_x={min_x}, max_x={max_x}, min_y={min_y}, max_y={max_y}")    
    
# Plot the largest cluster (likely the catenary)
plot_largest_cluster = lambda pcd: show_cloud(pcd, labels=np.zeros(pcd.shape[0]), downsample=10)
if largest_cluster is not None:
    plot_largest_cluster(largest_cluster)

