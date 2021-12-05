import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#https://stackabuse.com/classification-in-python-with-scikit-learn-and-pandas

NUMBER_OF_ITERATIONS = 200

print("Random Forest")
rawData = "./data/Embedding/tsne.csv"
rawDataFrame = pd.read_csv(rawData, index_col=0)

categoryColumns = list(rawDataFrame.filter(like='cat_').columns)

for column in categoryColumns:
	try:
		total = 0
		for i in range(NUMBER_OF_ITERATIONS):
			train, test = train_test_split(rawDataFrame, test_size=0.2)
			y_tr =  train[[column]]
			X_tr = train[[s for s in train.columns if "paraphrase-MiniLM-L6-v2_embedding_" in s]]

			y_test = test[[column]]
			X_test = test[[s for s in test.columns if "paraphrase-MiniLM-L6-v2_embedding_" in s]]

			RF = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
			RF.fit(X_tr, y_tr.values.ravel())
			RF.predict(X_test)
			total += RF.score(X_test,y_test)

		print(column[4:] + " - " + str(total/NUMBER_OF_ITERATIONS*100))
	except Exception as err:
		print(f'Error occurred In Column: {column[4:]}')
