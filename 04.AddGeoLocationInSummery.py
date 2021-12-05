import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
import requests
from requests.exceptions import HTTPError

#File Locations
countyData = "./data/county_data_processed.csv"
summeryData = "./data/summary.csv"
processedSummeryData = "./data/summary_processed.csv"

countyDataFrame = pd.read_csv(countyData, index_col=0)
summeryDataFrame = pd.read_csv(summeryData, index_col=0)

summeryDataFrame["latitude"] = ""	#countyDataFrame["latitude"]	#Throws duplicate error
summeryDataFrame["longitude"] = ""	#countyDataFrame["longitude"]

for index, row in summeryDataFrame.iterrows():
	try:
		summeryDataFrame.at[index, 'latitude'] = countyDataFrame.at[index, 'latitude']
		summeryDataFrame.at[index, 'longitude'] = countyDataFrame.at[index, 'longitude']
	except Exception as err:
		#summeryDataFrame.drop(summeryDataFrame.index[index], inplace=True)
		print(f'Error occurred in index- {index}: {err}')

#summeryDataFrame.set_index('CountyState', inplace=True)
summeryDataFrame.to_csv(processedSummeryData)
