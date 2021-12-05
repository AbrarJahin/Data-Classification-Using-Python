import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

rawData = "./data/Embedding/tsne.csv"
rawDataFrame = pd.read_csv(rawData, index_col=0)
plt.close("all")
sns.lmplot(x='tsne_x', y='tsne_y', data=rawDataFrame, hue='ColorCode', fit_reg=False)

plt.show()