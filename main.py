from sklearn.preprocessing import LabelEncoder

print("Email Spam Detection")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
data = pd.read_csv("dataset/emails.csv")
# print(data.head(5))
# print(data.shape)
# print(data.info)
# print(data.describe())
print(data.columns)

print(data["spam"])

print(data.isna().sum())
data = data.fillna("")
print(data.isna().sum())

LabelEncoding = LabelEncoder()

cleanData = pd.concat([data["text"], data["spam"]], axis=1)
#
# for column in cleanData.columns:
#     cleanData[column] = LabelEncoding.fit_transform(cleanData[column])


cleanData["spam"] = LabelEncoding.fit_transform(cleanData["spam"])
cleanData["text"] = LabelEncoding.fit_transform(cleanData["text"])

print(cleanData["text"])

print(cleanData.head(5))
regressor = LinearRegression()



xtrain, xtest, ytrain, ytest = train_test_split(cleanData["text"], data["spam"], test_size=20, random_state=40)

xtrain = xtrain.values.reshape(-1, 1)
ytrain = ytrain.values.reshape(-1, 1)
xtest = xtest.values.reshape(-1, 1)

regressor.fit(xtrain, ytrain)

predction = regressor.predict(xtest)
mse = mean_squared_error(ytest, predction)
print("mean_squared_error: ", mse)
