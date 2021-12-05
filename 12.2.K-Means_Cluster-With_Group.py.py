import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

rawData = "./data/Embedding/tsne.csv"
rawDataFrame = pd.read_csv(rawData, index_col=0)

kmeans = KMeans(n_clusters=17).fit(rawDataFrame[['tsne_x', 'tsne_y']])
centroids = kmeans.cluster_centers_

sns.lmplot(x='tsne_x', y='tsne_y', data=rawDataFrame, hue='ColorCode', fit_reg=False)
#rawDataFrame.plot(kind='scatter',x="tsne_x",y="tsne_y",c="ColorCode", s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)

#sns.lmplot(x='tsne_x', y='tsne_y', data=rawDataFrame, hue='ColorCode', fit_reg=False)

plt.show()
