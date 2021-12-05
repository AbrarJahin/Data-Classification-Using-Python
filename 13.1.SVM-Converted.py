import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm

NUMBER_OF_ITERATIONS = 1000

rawData = "./data/Embedding/tsne.csv"
rawDataFrame = pd.read_csv(rawData, index_col=0)

total = 0
for i in range(NUMBER_OF_ITERATIONS):

	train, test = train_test_split(rawDataFrame, test_size=0.2)

	y_tr =  train[['ColorCode']]
	X_tr = train[['tsne_x', 'tsne_y']]

	#y_test = test.iloc[:,0]
	#X_test = test.iloc[:,1:]
	y_test = test[['ColorCode']]
	X_test = test[['tsne_x', 'tsne_y']]

	SVM = svm.SVC(decision_function_shape="ovo").fit(X_tr, y_tr.values.ravel())
	SVM.predict(X_test)
	total += SVM.score(X_test, y_test)

print(total/NUMBER_OF_ITERATIONS*100)
