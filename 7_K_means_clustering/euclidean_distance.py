import numpy as np

a = np.array([1,2,3])
b = np.array([4,5,4])

euc_dis = np.sqrt(np.dot(b-a, b-a))
print(euc_dis)

# Other way 
euc_dis = np.linalg.norm(b - a)
print(euc_dis)

# Move Centroids logic
# 
X = np.array([[1,2], [2,3], [3,4], [4,5], [5,6]])
print(X)

cluster_group = np.array([1,0,0,0,1])

print(np.unique(cluster_group))

# for type in np.unique(cluster_group):
#     print(type)


# print(X[cluster_group==0])

print(f"New centroids: {X[cluster_group==0].mean(axis = 0)}")
print(f"New centroids: {X[cluster_group==1].mean(axis = 0)}")

for type in np.unique(cluster_group):
    print(X[cluster_group==type].mean(axis = 0))

