import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
import requests
from requests.exceptions import HTTPError

#File Locations
rawData = "./data/incidents.csv"
processedData = "./data/incidents_processed.csv"

rawDataFrame = pd.read_csv(rawData)
rawDataFrame["CountyState"] = ""
for index, row in rawDataFrame.iterrows():
	try:
		rawDataFrame.at[index, 'CountyState'] = rawDataFrame.at[index, 'county'] + " County " + rawDataFrame.at[index, 'state']
	except Exception as err:
		print(f'Error occurred during File Save: {err}')
rawDataFrame.to_csv(processedData)