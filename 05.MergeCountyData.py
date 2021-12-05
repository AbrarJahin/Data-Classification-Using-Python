import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
import requests
from requests.exceptions import HTTPError
from sklearn.preprocessing import OneHotEncoder
import datetime

#File Locations
rawData1 = "./data/mergedData_without_county.csv"
rawData2 = "./data/county_data_processed.csv"
mergedData = "./data/mergedData.csv"
mergedProcessedData = "./data/mergedProcessedData.csv"

rawDataFrame1 = pd.read_csv(rawData1, index_col=0)
rawDataFrame1 = rawDataFrame1[~rawDataFrame1.index.duplicated(keep='first')] #Remove Duplicates

rawDataFrame2 = pd.read_csv(rawData2, index_col=0)
rawDataFrame2 = rawDataFrame2[~rawDataFrame2.index.duplicated(keep='first')]

columnNames = list(rawDataFrame2.columns)
removable = ["FIPS", "county", "county_seat", "state"]
rawDataFrame2Columns = [elt for elt in columnNames if elt not in removable]

for index, row in rawDataFrame1.iterrows():
	try:
		df2index = rawDataFrame1.at[index, 'CountyState']
		for col in rawDataFrame2Columns:
			rawDataFrame1.at[index, col] = rawDataFrame2.at[df2index, col]
	except Exception as err:
		print(err, "-", df2index)

#Rename and del Columns
for i, colName in enumerate(rawDataFrame2Columns):
	if "(" in colName:
		newColName = colName.split("(")[0].strip().replace(" ", '_')
		print(colName, newColName)
		rawDataFrame1[newColName] = rawDataFrame1[colName]
		rawDataFrame1.drop([colName], axis = 1, inplace = True)

rawDataFrame1.to_csv(mergedData)

#Drop un-necessary Columns
columnsToDrop = ["State", "County", "City", "CountyState"]
rawDataFrame1.drop(columnsToDrop, axis=1, inplace=True)

def convertToYearDiff(x:str)->float:
	date_time_str = '2017-06-29 08:15:27.243860'
	date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S.%f')
	date = datetime.datetime.strptime(x, "%m/%d/%Y")
	return (date_time_obj-date).days/365.0

rawDataFrame1['Date'] = rawDataFrame1['Date'].apply(lambda x: convertToYearDiff(x))

def addY(x:str)->str:
	return 'Y_'+x

def processTypeString(x:str)->str:
	x = x.strip()
	x = x.replace(", ",",")
	x = x.replace(" ","_")
	xList = x.split(",")
	xList = list(map(addY, xList))
	x = ",".join(xList)
	return x

# Using One hot encoding for Type
rawDataFrame1['Type'] = rawDataFrame1['Type'].apply(lambda x: processTypeString(x))
rawDataFrame1 = pd.concat([rawDataFrame1.drop('Type', 1), rawDataFrame1['Type'].str.get_dummies(sep=",")], 1)

def processNatureString(x:str)->str:
	x = x.strip()
	x = x.replace("general antisemtism","general antisemitism")
	x = x.replace("conspiracy theory","conspiracy")
	x = x.replace("oppresion connection","oppression connection")
	x = x.replace(", ",",")
	x = x.replace(" ","_")
	return x

# Using One hot encoding for Nature
rawDataFrame1['Nature'] = rawDataFrame1['Nature'].apply(lambda x: processNatureString(x))
rawDataFrame1 = pd.concat([rawDataFrame1.drop('Nature', 1), rawDataFrame1['Nature'].str.get_dummies(sep=",")], 1)

def processLocationString(x:str)->str:
	x = x.strip()
	x = x.replace("off-campus","off campus")
	x = x.replace("off campus","outside school")
	x = x.replace(", ",",")
	x = x.replace(" ","_")
	return x

rawDataFrame1['Location'] = rawDataFrame1['Location'].apply(lambda x: processLocationString(x))
rawDataFrame1 = pd.concat([rawDataFrame1.drop('Location', 1), rawDataFrame1['Location'].str.get_dummies(sep=",")], 1)

def processPerpetratorString(x:str)->str:
	x = x.strip()
	x = x.replace(", ",",")
	x = x.replace(" ","_")
	xList = x.split(",")
	xList = ["Perpetrator-"+item for item in xList]
	x = ",".join(xList)
	return x

rawDataFrame1['Perpetrator'] = rawDataFrame1['Perpetrator'].apply(lambda x: processPerpetratorString(x))
rawDataFrame1 = pd.concat([rawDataFrame1.drop('Perpetrator', 1), rawDataFrame1['Perpetrator'].str.get_dummies(sep=",")], 1)
rawDataFrame1 = pd.concat([rawDataFrame1.drop('Target', 1), rawDataFrame1['Target'].str.get_dummies(sep=", ")], 1)

rawDataFrame1.to_csv(mergedProcessedData)
