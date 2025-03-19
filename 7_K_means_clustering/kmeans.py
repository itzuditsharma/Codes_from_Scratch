import random
import numpy as np
class Kmeans:
    def __init__(self, n_clusters = 2, max_iter = 100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit_predict(self, X):

        #Selecting random centroids this will be equal to n_clusters 
        random_index = random.sample(range(0, X.shape[0]), self.n_clusters)
        self.centroids = X[random_index]
        
        for _ in range(self.max_iter):
            # Assign clusters -> cluster_group will have [0,1,1,0,1] -> 1st row assigned to 0th cluster and 2nd row assigned to 1st cluster as so on 
            cluster_group = self.assign_clusters(X)
            old_centroids = self.centroids
            # Move centroids 
            self.centroids = self.move_centroids(X, cluster_group)
            # Check finish 
            if(old_centroids == self.centroids).all():
                break

        return cluster_group

    def assign_clusters(self, X):
        cluster_group = []
        distances = []
        # Calculate distance between every point and centroids and then select the one with min distance and assign it to that cluster 
        for row in X:
            for centroid in self.centroids:
                distances.append(np.sqrt(np.dot(row - centroid, row - centroid))) #Euclidean distance
            min_distance = min(distances)
            index_pos = distances.index(min_distance)
            cluster_group.append(index_pos)
            distances.clear()

        return np.array(cluster_group)
    
    # Separate the clusters based on 0's and 1's and then take means of them -> they will be the new centroids 
    def move_centroids(self, X, cluster_group):
        new_centroids = []

        cluster_type = np.unique(cluster_group)

        for type in cluster_type:
            new_centroids.append(X[cluster_group == type].mean(axis = 0))
        
        return np.array(new_centroids)
    