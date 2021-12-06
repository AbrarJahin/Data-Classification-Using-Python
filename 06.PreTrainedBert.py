import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
import requests
from requests.exceptions import HTTPError
from sentence_transformers import SentenceTransformer

#File Locations
rawData = "./data/mergedProcessedData.csv"
rawDataFrame = pd.read_csv(rawData, index_col=0)
bertData = "./data/Embedding/bert.csv"

#Normalize Data
'''
df = (df-df.min())/(df.max()-df.min())# min-max normalization:
df = (df-df.mean())/df.std()#mean normalization:
'''
#min-max normalization
rawDataFrame['date_norm'] = (rawDataFrame['Date']-rawDataFrame['Date'].min())/(rawDataFrame['Date'].max()-rawDataFrame['Date'].min())
rawDataFrame.drop(['Date'], axis = 1, inplace = True)

#mean normalization
rawDataFrame['population_2010'] = [int(str(item).replace(',', '')) for item in list(rawDataFrame['population_2010'])]
rawDataFrame['population_2010'] = (rawDataFrame['population_2010']-rawDataFrame['population_2010'].mean())/rawDataFrame['population_2010'].std()
rawDataFrame.drop(['population_2010'], axis = 1, inplace = True)

rawDataFrame['lat_norm'] = (rawDataFrame['latitude']-rawDataFrame['latitude'].mean())/rawDataFrame['latitude'].std()
rawDataFrame.drop(['latitude'], axis = 1, inplace = True)

rawDataFrame['lon_norm'] = (rawDataFrame['longitude']-rawDataFrame['longitude'].mean())/rawDataFrame['longitude'].std()
rawDataFrame.drop(['longitude'], axis = 1, inplace = True)

rawDataFrame['Land_Area'] = [float(str(item).replace(',', '')) for item in list(rawDataFrame['Land_Area'])]
rawDataFrame['l_area_norm'] = (rawDataFrame['Land_Area']-rawDataFrame['Land_Area'].mean())/rawDataFrame['Land_Area'].std()
rawDataFrame.drop(['Land_Area'], axis = 1, inplace = True)

rawDataFrame['Water_Area'] = [float(str(item).replace(',', '').replace('-', '')) if len(item)>2 else 0.0 for item in list(rawDataFrame['Water_Area'])]
rawDataFrame['w_area_norm'] = (rawDataFrame['Water_Area']-rawDataFrame['Water_Area'].mean())/rawDataFrame['Water_Area'].std()
rawDataFrame.drop(['Water_Area'], axis = 1, inplace = True)

rawDataFrame['Total_Area'] = [float(str(item).replace(',', '')) for item in list(rawDataFrame['Total_Area'])]
rawDataFrame['t_area_norm'] = (rawDataFrame['Total_Area']-rawDataFrame['Total_Area'].mean())/rawDataFrame['Total_Area'].std()
rawDataFrame.drop(['Total_Area'], axis = 1, inplace = True)

#Bert Apply in Description
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

for index, row in rawDataFrame.iterrows():
	try:
		embedding = model.encode(
				sentences=rawDataFrame.at[index, 'Description'],
				show_progress_bar=True,
				output_value = 'sentence_embedding', #'token_embeddings' 'sentence_embedding'
				normalize_embeddings = True
			)
		listEmbedding = list(embedding)
		for i, val in enumerate(listEmbedding):
			rawDataFrame.at[index, 'paraphrase-MiniLM-L6-v2_embedding_'+str(i)] = val
	except Exception as err:
		print(f'Error occurred during File Save: {err}')

rawDataFrame.drop(['Description'], axis = 1, inplace = True)
rawDataFrame.drop(['Fips_County'], axis = 1, inplace = True)

#Create Color Code to make it categorical
yColumns = list(rawDataFrame.filter(like='Y_').columns)
for index, row in rawDataFrame.iterrows():
	try:
		colorCode = 0
		for i, c in enumerate(yColumns):
			if rawDataFrame.at[index, c] != 0:
				colorCode+=2**i
		rawDataFrame.at[index, 'Color_Y'] = int(colorCode)
	except Exception as err:
		print(f'Error occurred during File Save: {err}')

rawDataFrame.to_csv(bertData)
