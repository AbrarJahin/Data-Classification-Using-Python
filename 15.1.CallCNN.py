from Cnn import Cnn
import pandas as pd
from sklearn.model_selection import train_test_split

def callCnnForTsne():
    rawData = "./data/Embedding/tsne.csv"
    rawDataFrame = pd.read_csv(rawData, index_col=0)
    rawDataFrame = rawDataFrame.astype(float)

    xColumns = list(rawDataFrame.filter(like='X').columns)

    train, test = train_test_split(rawDataFrame, test_size=0.2)
    y_tr =  train[['Color_Y']]
    X_tr = train[xColumns]

    y_test = test[['Color_Y']]
    X_test = test[xColumns]

    model = Cnn(X_tr, y_tr, X_test, y_test, epochs = 10, batch_size = 512)
    model.trainAndSaveModel()
    df = model.savePrediction(X_test)
    print("Done")

def callCnnForBert():
    rawData = "./data/Embedding/bert.csv"
    rawDataFrame = pd.read_csv(rawData, index_col=0)
    rawDataFrame = rawDataFrame.astype(float)

    #xColumns = list(rawDataFrame.filter(like='X').columns)
    yColumns = list(rawDataFrame.filter(like='Y_').columns)
    xColumns = list(set(rawDataFrame.columns).difference(list(yColumns)+["Color_Y"]))

    train, test = train_test_split(rawDataFrame, test_size=0.2)
    y_tr =  train[yColumns]
    X_tr = train[xColumns]

    y_test = test[yColumns]
    X_test = test[xColumns]

    model = Cnn(X_tr, y_tr, X_test, y_test, epochs = 10, batch_size = 512)
    model.trainAndSaveModel()
    df = model.savePrediction(X_test)
    print("Done")

callCnnForBert()
#callCnnForTsne()