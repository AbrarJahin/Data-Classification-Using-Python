import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
#https://stackabuse.com/classification-in-python-with-scikit-learn-and-pandas
#https://machinelearningmastery.com/multi-output-regression-models-with-python/

NUMBER_OF_ITERATIONS = 2

rawData = "./data/Embedding/tsne.csv"
rawDataFrame = pd.read_csv(rawData, index_col=0)

total = 0
for i in range(NUMBER_OF_ITERATIONS):
	train, test = train_test_split(rawDataFrame, test_size=0.2)
	#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

	#y_tr =  train[["bomb threat", "cyberbullying", "harassment", "literature dump", "not specified", "performance", "vandalism"]]
	y_tr =  train[["bomb threat", "cyberbullying", "harassment", "literature dump", "not specified", "performance", "vandalism"]]
	X_tr = train[[s for s in train.columns if "paraphrase-MiniLM-L6-v2_embedding_" in s]]

	y_test = test[["bomb threat", "cyberbullying", "harassment", "literature dump", "not specified", "performance", "vandalism"]]
	X_test = test[[s for s in test.columns if "paraphrase-MiniLM-L6-v2_embedding_" in s]]

	RF = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
	print(X_tr.shape, y_tr.shape)
	RF.fit(X_tr, y_tr.values.ravel())
	RF.predict(X_test)
	total += RF.score(X_test,y_test)

print(total/NUMBER_OF_ITERATIONS*100)
