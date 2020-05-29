import numpy as np
from sklearn.cluster import KMeans 

r = lambda: np.random.randint(1, 100)
X = [[r(), r()] for _ in range(50)]


kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
labels = kmeans.labels_

print(labels)