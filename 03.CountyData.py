import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
import requests
from requests.exceptions import HTTPError

#File Locations
rawData = "./data/county_data.csv"
processedData = "./data/county_data_processed.csv"
rawDataFrame = pd.read_csv(rawData, index_col=0)
#rawDataFrame['date']= pd.to_datetime(rawDataFrame['date'])
#processedDataFrame['state'] = processedDataFrame['state'].astype(str)
#processedDataFrame['county'] = processedDataFrame['county'].astype(str)
#processedDataFrame['city'] = processedDataFrame['city'].astype(str)
#processedDataFrame['CountyState'] = processedDataFrame['CountyState'].astype(str)
#processedDataFrame['description'] = processedDataFrame['description'].astype(str)
#print(processedDataFrame.dtypes)

summery = {}

rawDataFrame = pd.read_csv(rawData)
rawDataFrame["CountyState"] = ""
for index, row in rawDataFrame.iterrows():
	try:
		rawDataFrame.at[index, 'latitude'] = float(rawDataFrame.at[index, 'latitude'].strip().replace("–", "-")[:-1])
		rawDataFrame.at[index, 'longitude'] = float(rawDataFrame.at[index, 'longitude'].strip().replace("–", "-")[:-1])
		
		county = rawDataFrame.at[index, 'county'].replace(u'\xa0', u' ').strip()
		if county.find('[')>=0:
			county = county[:county.find("[")].strip()
		state = rawDataFrame.at[index, 'state'].replace(u'\xa0', u' ').strip()
		if state.find('[')>=0:
			state = state[:state.find("[")].strip()
		rawDataFrame.at[index, 'CountyState'] = county + " County " + state
	except Exception as err:
		rawDataFrame.drop(rawDataFrame.index[index], inplace=True)
		print(f'Error occurred during File Save: {err}')

rawDataFrame.set_index('CountyState', inplace=True)
del rawDataFrame['No']
rawDataFrame.to_csv(processedData)
