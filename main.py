print("Email Spam Detection")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("dataset/emails.csv")
# print(data.head(5))
# print(data.shape)
# print(data.info)
# print(data.describe())
print(data.columns)
print(data["spam"])

print(data.isna().sum())