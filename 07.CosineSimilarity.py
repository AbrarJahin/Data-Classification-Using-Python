import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
import requests
from requests.exceptions import HTTPError
from sentence_transformers import SentenceTransformer, util
import numpy

#File Locations
rawData = "./data/mergedData.csv"
rawDataFrame = pd.read_csv(rawData, index_col=0)
cosSimData = "./data/Embedding/Cosine-Similarity.csv"
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

adlIdList = list(rawDataFrame.index.values)
similarityDataFrame = pd.DataFrame(index=adlIdList, columns=adlIdList)

descriptions = rawDataFrame.loc[:,'Description'].values

descriptionEmbeddings = model.encode(descriptions)
cosineSimilarities = util.cos_sim(descriptionEmbeddings, descriptionEmbeddings)

all_sentence_combinations = []
for i in range(len(cosineSimilarities)-1):
    for j in range(i+1, len(cosineSimilarities)):
        all_sentence_combinations.append([cosineSimilarities[i][j], i, j])

#Sort list by the highest cosine similarity score
all_sentence_combinations = sorted(all_sentence_combinations, key=lambda x: x[0], reverse=True)

for score, i, j in all_sentence_combinations[:]:
    #similarityDataFrame['Column'].iloc[0]
    print(i,j)
    similarityDataFrame[adlIdList[i]].iloc[j] = score
similarityDataFrame.to_csv(cosSimData)