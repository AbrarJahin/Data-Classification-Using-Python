import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm

NUMBER_OF_ITERATIONS = 10

rawData = "./data/Embedding/tsne.csv"
rawDataFrame = pd.read_csv(rawData, index_col=0)
xColumns = list(rawDataFrame.filter(like='X').columns)
total = 0
for i in range(NUMBER_OF_ITERATIONS):

	train, test = train_test_split(rawDataFrame, test_size=0.2)

	y_tr =  train[['Color_Y']]
	X_tr = train[xColumns]

	y_test = test[['Color_Y']]
	X_test = test[xColumns]

	#clf = svm.SVC(decision_function_shape='ovo')
	#clf.fit(y_tr, y_tr)
	#clf.predict(X_test)

	SVM = svm.SVC(decision_function_shape="ovo").fit(X_tr, y_tr.values.ravel())	#.values.ravel()
	SVM.predict(X_test)
	total += SVM.score(X_test, y_test)

print(total/NUMBER_OF_ITERATIONS*100)
