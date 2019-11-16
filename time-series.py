


import pandas
import numpy as np
from matplotlib import pyplot as plt
import math
from sklearn.neural_network import MLPRegressor


# Read Data in Dataframe
# Entire the training data file here
data = (pandas.read_excel('project2_train.xlsx', header=None))
index_data = data[0].index.values

# Function to calculate MSE

def mse(y_test, y_pred):
    error = 0
    for j in range(0, len(y_pred)-1):
        error += (y_test[j] - y_pred[j]) ** 2

    mse = math.sqrt(error / len(y_pred))
    return mse

# Function to predict the values of the test input

def predict(x_train, x_test_input, model):
    y_pred = []

    for i in range(0,len(x_test_input)):
        x_temp = []
        x_test = []
        if (len(y_pred) < 3):
            x_temp.append(data[0][len(x_train)-3])
            x_temp.append(data[0][len(x_train)-2])
            x_temp.append(data[0][len(x_train)-1])
        else:
            x_temp.append(y_pred[i - 3])
            x_temp.append(y_pred[i - 2])
            x_temp.append(y_pred[i - 1])
        x_test.append(x_temp)

        # Predict with the trained model

        y_pred_itermed = model.predict(x_test)
        y_pred.append(y_pred_itermed[0])
    return y_pred

# Function to train the model

def train(x_train, y_train, number_hidden_nodes):


    neural_model = MLPRegressor(hidden_layer_sizes=(number_hidden_nodes,), activation='tanh', solver='adam', alpha=0.001, max_iter=2000, learning_rate_init=0.005,
                      learning_rate='adaptive', early_stopping=True, shuffle=True, random_state=9, momentum=0.9,
                      nesterovs_momentum=True, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    n = neural_model.fit(x_train, y_train)

    return n

# Data slicing in to train and test

def data_slicing(data, trainig_size, test_size):
    x_train = []
    y_train = []
    for i in range(0, trainig_size):
        x_temp = []
        for j in range(0,3):
            x_temp.append(data[0][i+j])
        x_train.append(x_temp)
        y_train.append(data[0][i+3])

    x_test = []
    y_test = []
    for j in range(trainig_size, trainig_size + test_size):
        x_test.append(j)
        y_test.append(data[0][j])

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':

    # Slice input data

    x_train, y_train, x_test, y_test = data_slicing(data, 275, 30)

    # Get the trained model

    model = train(x_train, y_train, 8)

    # Predict the output

    y_pred = predict(x_train,x_test, model)

    # Calulate the error for the test values

    error = mse(y_test, y_pred)


    print("Error", error)

    print("Predicted test output", y_pred)

