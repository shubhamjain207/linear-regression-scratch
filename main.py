import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split



pd = pd.read_csv("insurance_data.csv")

y = pd.bought_insurance
x = pd[['age','affordibility']]


x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=25)

x_train_scaled = x_train.copy()
x_train_scaled['age'] = x_train_scaled['age'] / 100


x_test_scaled = x_test.copy()
x_test_scaled['age'] = x_test_scaled['age'] / 100




model = keras.Sequential([

    keras.layers.Dense(1, input_shape=(2,),activation='sigmoid',kernel_initializer='ones',bias_initializer='zeros')

])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(x_train_scaled, y_train, epochs=5000)
model.evaluate(x_test_scaled, y_test)

print(x_test_scaled)

print(model.predict(x_test_scaled))

coef, intercept  = model.get_weights()


# def prediction_function(age, affordibility):
#     weighted_sum = coef[0]* age + coef[1] * affordibility + intercept
#     return 1 / (1 + math.exp(-weighted_sum))
#
# print("Custom ", prediction_function(0.29, 0))


def gradient_descent(age, affordibility, y_true, epochs):
    w1 = 1
    w2 = 1
    bias = 0
    learning  = 0.5
    n = len(age)

    for i in range(epochs):
        weighted_sum = w1 * age + w2 * affordibility + bias
        y_predicted = 1 / (1 + math.exp(-weighted_sum));

gradient_descent(x_train_scaled['age'], x_train_scaled['affordibility'], y_train, 1000)



