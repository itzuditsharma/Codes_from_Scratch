import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class Kmeans:
    def __init__(self, n_clusters=2, max_iter=200):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
    
    def fit_predict(self, X):
        random_index = random.sample(range(0, X.shape[0]), self.n_clusters)
        self.centroids = X[random_index]
        
        for _ in range(self.max_iter):
            cluster_group = self.assign_clusters(X)
            old_centroids = self.centroids
            self.centroids = self.move_centroids(X, cluster_group)
            
            if (old_centroids == self.centroids).all():
                break
        
        return cluster_group

    def assign_clusters(self, X):
        cluster_group = []
        
        for row in X:
            distances = [np.linalg.norm(row - centroid) for centroid in self.centroids]
            cluster_group.append(np.argmin(distances))
        
        return np.array(cluster_group)
    
    def move_centroids(self, X, cluster_group):
        new_centroids = []
        
        for type in np.unique(cluster_group):
            new_centroids.append(X[cluster_group == type].mean(axis=0))
        
        return np.array(new_centroids)
    
    def compute_wcss(self, X):
        cluster_group = self.fit_predict(X)
        wcss = 0
        for i in range(self.n_clusters):
            cluster_points = X[cluster_group == i]
            wcss += np.sum((cluster_points - self.centroids[i]) ** 2)
        return wcss
    
    @staticmethod
    def find_optimal_k(X, max_k=10):
        wcss_values = []
        k_values = range(1, max_k + 1)
        
        for k in k_values:
            kmeans = Kmeans(n_clusters=k)
            wcss_values.append(kmeans.compute_wcss(X))
        
        plt.plot(k_values, wcss_values, marker='o')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
        plt.title('Elbow Method for Optimal K')
        plt.show()


X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)
    
# Apply K-Means clustering
kmeans = Kmeans(n_clusters=3, max_iter=100)
cluster_labels = kmeans.fit_predict(X)

# Visualize the clustering results
plt.figure(figsize=(8, 6))
for i in range(kmeans.n_clusters):
    plt.scatter(X[cluster_labels == i, 0], X[cluster_labels == i, 1], label=f'Cluster {i}')

# Plot centroids
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
            s=200, c='black', marker='X', edgecolors='white', label='Centroids')

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering Results")
plt.legend()
plt.show()

# Finding the optimal number of clusters using the Elbow method
Kmeans.find_optimal_k(X, max_k=10)