import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd

# Visualizing - https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b

#Working Procedure - https://medium.com/dunder-data/from-pandas-to-scikit-learn-a-new-exciting-workflow-e88e2271ef62

#PSA

X_EMBEDDING_COLUMN_COUNT = 1
Y_EMBEDDING_COLUMN_COUNT = 1

rawData = "./data/Embedding/bert.csv"
rawDataFrame = pd.read_csv(rawData, index_col=0)
rawDataFrame = rawDataFrame.astype(float)
pcaData = "./data/Embedding/pca.csv"

yColumns = list(rawDataFrame.filter(like='Y_').columns)
xColumns = list(set(rawDataFrame.columns).difference(list(yColumns)+["Color_Y"]))

tsneDf = pd.DataFrame()

######Save X values
pcaX = PCA(n_components=X_EMBEDDING_COLUMN_COUNT).fit(rawDataFrame[xColumns])
for i in range(X_EMBEDDING_COLUMN_COUNT):
    tsneDf["X"+str(i)] = list(tsneMbeddedX[:, i])

######Save Y values
tsneMbeddedY = TSNE(n_components=Y_EMBEDDING_COLUMN_COUNT).fit_transform(rawDataFrame[yColumns])
for i in range(tsneMbeddedY.shape[1]):
    tsneDf["Y"+str(i)] = list(tsneMbeddedY[:, i])

tsneDf.to_csv(pcaData)
