import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
import requests
from requests.exceptions import HTTPError

#File Locations
rawData = "./data/incidents_processed.csv"
summeryData = "./data/summary.csv"
rawDataFrame = pd.read_csv(rawData, index_col=0)
rawDataFrame['date']= pd.to_datetime(rawDataFrame['date'])
#processedDataFrame['state'] = processedDataFrame['state'].astype(str)
#processedDataFrame['county'] = processedDataFrame['county'].astype(str)
#processedDataFrame['city'] = processedDataFrame['city'].astype(str)
#processedDataFrame['CountyState'] = processedDataFrame['CountyState'].astype(str)
#processedDataFrame['description'] = processedDataFrame['description'].astype(str)
#print(processedDataFrame.dtypes)

summery = {}

for index, row in rawDataFrame.iterrows():
	try:
		#rawDataFrame.at[index, 'CountyState'] = rawDataFrame.at[index, 'county'] + " County " + rawDataFrame.at[index, 'state']
		key = rawDataFrame.at[index, 'CountyState'].strip()
		description = rawDataFrame.at[index, 'description'].strip()
		if key in summery and description not in summery[key]['descriptions']:
			summery[key]['no_of_events']+=1
			summery[key]['cities'].add(rawDataFrame.at[index, 'city'])
			summery[key]['descriptions'].add(description)
			#summery[key]['city_count']+=1
			summery[key]['first_occurrence'] = min(summery[key]['first_occurrence'], rawDataFrame.at[index, 'date'])
			summery[key]['last_occurrence'] = max(summery[key]['last_occurrence'], rawDataFrame.at[index, 'date'])
		else:
			summery[key] = {}
			summery[key]['descriptions'] = { description }
			summery[key]['no_of_events'] = 1
			summery[key]['cities'] = { rawDataFrame.at[index, 'city'] }
			summery[key]['first_occurrence'] = rawDataFrame.at[index, 'date']
			summery[key]['last_occurrence'] = rawDataFrame.at[index, 'date']
		summery[key]["duration"] = summery[key]['last_occurrence'] - summery[key]['first_occurrence']
		summery[key]["avg_interval"] = summery[key]["duration"]/summery[key]['no_of_events'] if summery[key]['no_of_events']>1 else float('inf')
	except Exception as err:
		print(f'Error occurred during File Save: {err}')
summeryDataFrame = pd.DataFrame(
							{'CountyState':list(summery.keys())},
							columns=['CountyState'],
							#index='CountyState'
						)
summeryDataFrame.set_index('CountyState', inplace=True)
for key in summery:
	summeryDataFrame.at[key, 'no_of_events'] = summery[key]['no_of_events']
	summeryDataFrame.at[key, 'first_occurrence'] = summery[key]['first_occurrence']
	summeryDataFrame.at[key, 'last_occurrence'] = summery[key]['last_occurrence']
	summeryDataFrame.at[key, 'duration'] = summery[key]['duration']
	summeryDataFrame.at[key, 'avg_interval'] = summery[key]['avg_interval']
	summeryDataFrame.at[key, 'cities'] = ', '.join(sorted(list(summery[key]['cities']), reverse=False))
	summeryDataFrame.at[key, 'descriptions'] = sorted(list(summery[key]['descriptions']), reverse=False)
summeryDataFrame.sort_values(by='no_of_events', ascending=False, inplace=True)
summeryDataFrame.to_csv(summeryData)
