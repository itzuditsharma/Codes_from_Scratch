import numpy as np
import random

class Kmeans:
    def __init__(self, n_clusters = 2, max_iter = 100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit_predict(self, X):
        random_idx = random.sample(range(X.shape[0]), self.n_clusters)
        self.centroids = X[random_idx]

        for _ in range(self.max_iter):
            cluster_group = self.assign_clusters(X)
            old_centroids = self.centroids

            self.centroids = self.move_centroids(X, cluster_group)
            if(old_centroids == self.centroids).all():
                break

        return cluster_group

    def assign_clusters(self, X):
        cluster_groups = []
        distances = []

        for row in X:
            for centroid in self.centroids:
                distances.append(np.sqrt(np.dot(row - centroid, row - centroid)))
            min_distance = min(distances)
            index_pos = distances.index(min_distance)
            cluster_groups.append(index_pos)
            distances.clear()

        return np.array(cluster_groups)

    def move_centroids(self, X, cluster_group):
        new_centroids = []
        cluster_type = np.unique(cluster_group)

        for type in cluster_type:
            new_centroids.append(X[cluster_group == type].mean(axis = 0))

        return np.array(new_centroids)
    
    
  



        

