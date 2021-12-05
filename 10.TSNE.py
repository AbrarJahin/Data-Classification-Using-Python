import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd

# Visualizing - https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b

#Working Procedure - https://medium.com/dunder-data/from-pandas-to-scikit-learn-a-new-exciting-workflow-e88e2271ef62

EMBEDDING_COLUMN_COUNT = 384

rawData = "./data/Embedding/bert-withSegmentation.csv"
rawDataFrame = pd.read_csv(rawData, index_col=0)
tsneData = "./data/Embedding/tsne.csv"

#Example##########################################
#X = np.random.rand(2,384)
#X_mbedded = TSNE(n_components=2).fit_transform(X)
##################################################

#dfIds = rawDataFrame.values[:,0]
#a = rawDataFrame[rawDataFrame.columns[-EMBEDDING_COLUMN_COUNT:]].to_numpy()
#X_mbedded = TSNE(n_components=2).fit_transform(a)

matrix = np.empty(shape=[0, EMBEDDING_COLUMN_COUNT])

for index, row in rawDataFrame.iterrows():
	try:
		#embeddedRow = rawDataFrame.iloc[index, -EMBEDDING_COLUMN_COUNT:]
		npArray = row[-EMBEDDING_COLUMN_COUNT:].to_numpy()
		matrix = np.append(matrix, np.array([npArray.tolist()]), axis=0)
	except Exception as err:
		print(f'Error occurred during Matrix Processing: {err}')

tsneMbedded = TSNE(n_components=2).fit_transform(matrix)
curIndex = 0

for index, row in rawDataFrame.iterrows():
	try:
		rawDataFrame.at[index, "tsne_x"] = tsneMbedded[curIndex][0]
		rawDataFrame.at[index, "tsne_y"] = tsneMbedded[curIndex][1]
		curIndex+=1
	except Exception as err:
		print(f'Error occurred during File Save: {err}')
rawDataFrame.to_csv(tsneData)
