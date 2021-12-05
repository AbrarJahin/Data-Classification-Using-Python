import pandas as pd
import matplotlib.pyplot as plt

rawData = "./data/Embedding/tsne.csv"
rawDataFrame = pd.read_csv(rawData, index_col=0)
plt.close("all")
############################################################
groups = set()
for index, row in rawDataFrame.iterrows():
	try:
		groups.add(rawDataFrame.at[index, 'ColorCode'])
	except Exception as err:
		print(f'Error occurred during File Save: {err}')
############################################################
rawDataFrame.plot(kind='scatter',x="tsne_x",y="tsne_y",c="ColorCode")	#Automatic from equal distribution
print(len(groups), groups)
plt.show()