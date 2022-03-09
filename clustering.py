# Import modules and packages
import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans

# Read data
data = pd.read_csv('https://raw.githubusercontent.com/sowmyacr/kmeans_cluster/master/CLV.csv')
X = np.array(data)

# System constants
number_of_clusters = 4

# Apply KMEans to the Data
kmeans = KMeans(
	n_clusters=4,
	init='k-means++',
	max_iter=300,
	n_init=10,
	random_state=19890528,
	precompute_distances=True)

kfit = kmeans.fit(X)

freeze_centroids = kmeans.cluster_centers_
print(freeze_centroids)
print(freeze_centroids.shape)

# Save centroids as pickle file locally
with open('freezed_centroids.pkl','wb') as f:
    pickle.dump(freeze_centroids, f)
    print('Centroids are being saved to pickle file.')