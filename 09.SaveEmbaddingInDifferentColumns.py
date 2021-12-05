import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
import requests
from requests.exceptions import HTTPError
from sentence_transformers import SentenceTransformer

#File Locations
rawData = "./data/category-segmentation.csv"
rawDataFrame = pd.read_csv(rawData, index_col=0)
#prevColumnCount = len(rawDataFrame.columns)
bertData = "./data/Embedding/bert-withSegmentation.csv"

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

for index, row in rawDataFrame.iterrows():
	try:
		embedding = model.encode(rawDataFrame.at[index, 'Description'])
		for embadeIndex, val in enumerate(embedding):
			rawDataFrame.at[index, "paraphrase-MiniLM-L6-v2_embedding_" + str(embadeIndex).zfill(3)] = val
		#rawDataFrame.at[index, 'Embadding'] = embeddingString
	except Exception as err:
		print(f'Error occurred during File Save: {err}')
rawDataFrame.to_csv(bertData)
