import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sx as sx
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

sy = preprocessing.MinMaxScaler()

df = pd.read_csv("Housing.csv")

x = df.drop('price', axis='columns')
y = df['price']

weight1 = 1
weight2 = 1
weight3 = 1
weight4 = 1
weight5 = 1

bias = 0
list = []

x1 = x.values.tolist()
y1 = y.values.tolist()
predicted_value_list = []

meanSum = 0

for i in range(len(x1)):
    for j in range(len(x1[i])):
        list.append(x1[i][j])
    predicted_value = ((list[0] * weight1) + (list[1] * weight2) + (list[2] * weight3) + (list[3] * weight4) + (
                list[4] * weight5)) - bias
    predicted_value_list.append(predicted_value)
    list.clear()

for i in range(len(x1)):
    meanSum += pow((y1[i] - predicted_value_list[i]), 2)


meanSum /= len(x1)





print(meanSum)
