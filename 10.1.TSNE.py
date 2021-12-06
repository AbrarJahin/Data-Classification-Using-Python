import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd

# Visualizing - https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b

#Working Procedure - https://medium.com/dunder-data/from-pandas-to-scikit-learn-a-new-exciting-workflow-e88e2271ef62

X_EMBEDDING_COLUMN_COUNT = 20
Y_EMBEDDING_COLUMN_COUNT = 1

rawData = "./data/Embedding/bert.csv"
rawDataFrame = pd.read_csv(rawData, index_col=0)
rawDataFrame = rawDataFrame.astype(float)
tsneData = "./data/Embedding/tsne.csv"

yColumns = list(rawDataFrame.filter(like='Y_').columns)
xColumns = list(set(rawDataFrame.columns).difference(list(yColumns)+["Color_Y"]))

tsneDf = pd.DataFrame(columns = ['adl_id'])
tsneDf['adl_id'] = list(rawDataFrame.index)
tsneDf = tsneDf.set_index('adl_id')

######Save X values
tsneMbeddedX = TSNE(method='exact', n_components=X_EMBEDDING_COLUMN_COUNT).fit_transform(rawDataFrame[xColumns])
for i in range(tsneMbeddedX.shape[1]):
    tsneDf["X"+str(i)] = list(tsneMbeddedX[:, i])

columnNames = tsneDf.columns.to_list()

for c in columnNames:
    #df = (df-df.mean())/df.std()#mean normalization:
    tsneDf[c] = (tsneDf[c]-tsneDf[c].mean())/tsneDf[c].std()

######Save Y values
#tsneMbeddedY = TSNE(n_components=Y_EMBEDDING_COLUMN_COUNT).fit_transform(rawDataFrame[yColumns])
#for i in range(tsneMbeddedY.shape[1]):
#    tsneDf["Y"+str(i)] = list(tsneMbeddedY[:, i])

#Create Color Code to make it categorical
#for index, row in rawDataFrame.iterrows():
#	try:
#		colorCode = 0
#		for i, c in enumerate(yColumns):
#			if rawDataFrame.at[index, c] != 0:
#				colorCode+=2**i
#		tsneDf.at[index, 'Y_Color'] = int(colorCode)
#	except Exception as err:
#		print(f'Error occurred during File Save: {err}')

tsneDf['Color_Y'] = rawDataFrame['Color_Y']

tsneDf.to_csv(tsneData)
