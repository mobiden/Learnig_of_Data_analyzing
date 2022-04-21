import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)

X, y = make_blobs(n_samples=150, n_features=2, centers=4, random_state=1)
#plt.scatter(X[:, 0], X[:, 1])
#plt.show()
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=1)
kmeans.fit(X)
labels = kmeans.labels_
#plt.scatter(X[:, 0], X[:, 1], c=labels) #количество кластеров - 2
#plt.show()

crit = []
for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(X)
    crit.append(kmeans.inertia_)
# x`plt.plot(range(2,8), crit) # график локтя. оптимальное число - 4
for r in range(1, 6):
    kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=r)
    kmeans.fit(X)
    labels = kmeans.labels_
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.show()