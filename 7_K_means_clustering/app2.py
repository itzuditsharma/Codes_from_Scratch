from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from kmean_prac import Kmeans
import pandas as pd


# centroids = [(-5, -5), (5,5), (-2.5, 2.5)]
# cluster_std = [1,1,1]

# X,y = make_blobs(n_samples=100, cluster_std=cluster_std, centers=centroids, n_features=2, random_state=10)

df = pd.read_csv('student_clustering.csv')
print(df.shape)
X = df.iloc[:,:].values

km = Kmeans(n_clusters=4, max_iter=1000)
y_means= km.fit_predict(X)

plt.scatter(X[y_means == 0, 0], X[y_means==0,1], color = 'red')
plt.scatter(X[y_means == 1, 0], X[y_means==1,1], color = 'blue')

plt.show()
