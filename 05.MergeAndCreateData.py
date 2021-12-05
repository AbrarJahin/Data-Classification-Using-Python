import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
import requests
from requests.exceptions import HTTPError

#File Locations
rawData1 = "./data/incidents.csv"
rawData2 = "./data/code-reconciled.csv"
mergedData = "./data/mergedData_without_county.csv"

rawDataFrame1 = pd.read_csv(rawData1, index_col=5)
rawDataFrame1 = rawDataFrame1[~rawDataFrame1.index.duplicated(keep='first')] #Remove Duplicates
rawDataFrame1['date'] = pd.to_datetime(rawDataFrame1['date'])

rawDataFrame2 = pd.read_csv(rawData2, index_col=0)
rawDataFrame2.rename({'code.type': 'Type', 'code.nature': 'Nature', 'code.location': 'Location', 'code.perpetrator': 'Perpetrator', 'code.target': 'Target', 'description': 'Description'}, axis=1, inplace=True)

for index, row in rawDataFrame2.iterrows():
	try:
		#rawDataFrame1.drop(rawDataFrame1.index[[1,3]], inplace=True)
		#rawDataFrame1.drop(index = 2, inplace = True)
		if index in rawDataFrame1.index:
			rawDataFrame2.at[index, 'State'] = rawDataFrame1.at[index, 'state']
			rawDataFrame2.at[index, 'County'] = rawDataFrame1.at[index, 'county']
			rawDataFrame2.at[index, 'City'] = rawDataFrame1.at[index, 'city']
			rawDataFrame2.at[index, 'Fips_County'] = rawDataFrame1.at[index, 'fips_county']
			rawDataFrame2.at[index, 'Date'] = rawDataFrame1.at[index, 'date']
			rawDataFrame2.at[index, 'CountyState'] = rawDataFrame1.at[index, 'county'] + " County " + rawDataFrame1.at[index, 'state']
		else:
			rawDataFrame2.drop(index = index, inplace = True)
	except Exception as err:
		rawDataFrame2.drop(index = index, inplace = True)
		print(f'Error occurred during File Save: {err}')
rawDataFrame2.to_csv(mergedData)
