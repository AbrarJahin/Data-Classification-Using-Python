import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
#https://stackabuse.com/classification-in-python-with-scikit-learn-and-pandas

NUMBER_OF_ITERATIONS = 10

print("KNN")
rawData = "./data/Embedding/bert.csv"
rawDataFrame = pd.read_csv(rawData, index_col=0)
rawDataFrame = rawDataFrame.astype(float)

yColumns = list(rawDataFrame.filter(like='Y_').columns)
xColumns = list(set(rawDataFrame.columns).difference(list(yColumns)+['Color_Y']))

try:
	total = 0
	for i in range(NUMBER_OF_ITERATIONS):
		train, test = train_test_split(rawDataFrame, test_size=0.2)
		y_tr =  train[yColumns]
		X_tr = train[xColumns]

		y_test = test[yColumns]
		X_test = test[xColumns]

		neigh = NearestNeighbors(n_neighbors=len(yColumns))
		neigh.fit(X_tr)
		y = neigh.kneighbors_graph(X_test).toarray()
		total += RF.score(X_test,y_test)
	print(str(total/NUMBER_OF_ITERATIONS*100))
except Exception as err:
	print(f'Error occurred In Column: {column[4:]}')
