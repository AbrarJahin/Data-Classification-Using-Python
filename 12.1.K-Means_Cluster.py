import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
#https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203

rawData = "./data/Embedding/tsne.csv"
rawDataFrame = pd.read_csv(rawData, index_col=0)

kmeans = KMeans(n_clusters=4).fit(rawDataFrame[['tsne_x', 'tsne_y']])
centroids = kmeans.cluster_centers_

plt.scatter(rawDataFrame['tsne_x'], rawDataFrame['tsne_y'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()
