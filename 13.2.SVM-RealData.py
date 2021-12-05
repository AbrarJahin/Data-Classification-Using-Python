import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
#https://stackabuse.com/classification-in-python-with-scikit-learn-and-pandas

NUMBER_OF_ITERATIONS = 100

rawData = "./data/Embedding/tsne.csv"
rawDataFrame = pd.read_csv(rawData, index_col=0)

total = 0
for i in range(NUMBER_OF_ITERATIONS):

	train, test = train_test_split(rawDataFrame, test_size=0.2)

	#y_tr =  train[["bomb threat", "cyberbullying", "harassment", "literature dump", "not specified", "performance", "vandalism"]]
	y_tr =  train[['ColorCode']]
	X_tr = train[[s for s in train.columns if "paraphrase-MiniLM-L6-v2_embedding_" in s]]

	y_test = test[['ColorCode']]
	X_test = test[[s for s in test.columns if "paraphrase-MiniLM-L6-v2_embedding_" in s]]

	SVM = svm.SVC(decision_function_shape="ovo").fit(X_tr, y_tr.values.ravel())
	SVM.predict(X_test)
	total += SVM.score(X_test, y_test)

	##################


print(total/NUMBER_OF_ITERATIONS*100)
