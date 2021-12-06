import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
#https://stackabuse.com/classification-in-python-with-scikit-learn-and-pandas

print("SVM")

NUMBER_OF_ITERATIONS = 1

rawData = "./data/Embedding/bert.csv"
rawDataFrame = pd.read_csv(rawData, index_col=0)
rawDataFrame = rawDataFrame.astype(float)

yColumns = list(rawDataFrame.filter(like='Y_').columns)
xColumns = list(set(rawDataFrame.columns).difference(list(yColumns)))

categoryColumns = list(rawDataFrame.filter(like='cat_').columns)

try:
	total = 0
	for i in range(NUMBER_OF_ITERATIONS):
		train, test = train_test_split(rawDataFrame, test_size=0.2)
		y_train =  train[yColumns]
		X_train = train[xColumns]

		y_test = test[yColumns]
		X_test = test[xColumns]

		##############
		linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(X_train, y_train)
		rbf = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo').fit(X_train, y_train)
		poly = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo').fit(X_train, y_train)
		sig = svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo').fit(X_train, y_train)
		##############

		SVM = svm.SVC(decision_function_shape="ovo")
		SVM.fit(X_tr, y_tr)
		SVM.predict(X_test)
		total += SVM.score(X_test, y_test)

	print("Score - " + str(total/NUMBER_OF_ITERATIONS*100))
except Exception as err:
	print(f'Error occurred In Column: {column[4:]}')
