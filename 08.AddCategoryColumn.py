import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
import requests
from requests.exceptions import HTTPError
from sentence_transformers import SentenceTransformer

groups = set()

def GetGroupColorCode(sortedCategories, categories) -> int:
	calculatedColorCode, curValToAdd, catIndex = 0, 1, 0
	for i in range(len(categories)):
		categories[i] = "cat_" + categories[i].lstrip()
	#categories.sort()
	for c in sortedCategories:
		if c in categories:
			calculatedColorCode+=curValToAdd
		curValToAdd *= 2
	groups.add(calculatedColorCode)
	return calculatedColorCode

#File Locations
rawData = "./data/mergedData.csv"
rawDataFrame = pd.read_csv(rawData, index_col=0)
#rawDataFrame.set_index('adl_id', inplace=True)
categoryData = "./data/category-segmentation.csv"

categorySet = set()
for index, row in rawDataFrame.iterrows():
	try:
		categories = rawDataFrame.at[index, 'Type'].split(",")
		for category in categories:
			categorySet.add("cat_" + category.lstrip())
	except Exception as err:
		print(f'Error occurred during File Save: {err}')
sortedCategories = sorted(list(categorySet))

for category in sortedCategories:
	rawDataFrame[category] = False

for index, row in rawDataFrame.iterrows():
	try:
		categories = rawDataFrame.at[index, 'Type'].split(",")
		for category in categories:
			#rawDataFrame.at[index, category.lstrip()] = True
			rawDataFrame.loc[index, "cat_" + category.lstrip()] = True
		rawDataFrame.at[index, 'ColorCode'] = GetGroupColorCode(sortedCategories, categories)
	except Exception as err:
		print(f'Error occurred during File Save: {err}')

rawDataFrame.to_csv(categoryData)
print(sorted(list(groups)))
print("Total No Of Groups - " + str(len(groups)))