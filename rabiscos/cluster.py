#%% Generate sample data
from itertools import cycle

from sklearn.cluster import estimate_bandwidth, MeanShift
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

centers = [[1, 1], [-.75, -1], [1, -1], [-3, 2]]
X, _ = make_blobs(n_samples=5000, centers=centers, cluster_std=0.6)

#%% Compute clustering with MeanShift

# The bandwidth can be automatically estimated
bandwidth = estimate_bandwidth(X, quantile=.1,n_samples=500)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

n_clusters_ = labels.max()+1

#%% Plot result
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1],
    'o', markerfacecolor=col,
    markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()