# Code for Homework 3 ===========================================================
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from random import seed
from numpy import *
from sklearn.preprocessing import StandardScaler

dataset = pd.read_excel ('HW3train.xlsx')
data = dataset.values

dataset2 = pd.read_excel('HW3Validate.xlsx')
data2 = dataset2.values

# Split into inputs and outputs
inputs = data[0:256, 0:2].astype(float)
outputs = data[0:256:, 2]

inputs2 = data2[:, 0:2].astype(float)
outputs2 = data2[:, 2]

outputs2 = np.array(outputs2).reshape((-1,1))
outputs = np.array(outputs).reshape((-1,1))

# Activation functions and their derivatives
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def der_sigmoid(x):
    return sigmoid(x)*(1.0-sigmoid (x))

def ReLu(x):
    return np.maximum(x, 0)

def der_ReLu(x):
    x[x >= 0] = 1
    x[x < 0] = 0
    return x   

# Scaling Data
std_scaler = StandardScaler()
std_scaler.fit(inputs)
inputs = std_scaler.transform(inputs)

# Initialize and create hidden layers
w1 = np.random.rand(len(inputs[0]), 10) # weights for input -> h(0) hidden layer
w2 = np.random.rand(10,10) # weights for h(0) -> h(1) hidden layer
wo = np.random.rand(10,1) # weights for h(1) -> output layer 

# Initialize learning rate
learningRate = 0.01 # between 0.1 and 10^-6

# Initialize the biases with random uniform values
bf1 = np.random.rand(len(inputs), 10)
bf2 = np.random.rand(len(inputs), 10)
bf3 = np.random.rand(len(inputs), 1)

# Scale biases
s1 = StandardScaler()
s1.fit(bf1)
b1 = s1.transform(bf1)

s2 = StandardScaler()
s2.fit(bf2)
b2 = s2.transform(bf2)


s3 = StandardScaler()
s3.fit(bf3)
b3 = s3.transform(bf3)


#-------------------------------------------------------------------------#
# Train set
loss = []
for i in range (800):
    
    # Forward propagation
    z1 = np.dot(inputs, w1) + b1 # dot product for input -> h(0)
    h1 = ReLu(z1)   # activation function
    
    z2 = np.dot(h1, w2) + b2 # dot product for h(0) -> h(1)
    h2 = ReLu(z2) 
    
    zo = np.dot(h2, wo) + b3 # dot product for h(1) -> output
    ho = sigmoid(zo) # output
    
    # Define cost function
    cost = ((1 / 2) * (np.power((ho - outputs), 2)))
    loss.append(cost[0])
 
    # Back-propagation
    
    # Output layer
    dcost_dho = ho - outputs
    dho_dzo = der_sigmoid(zo) 
    dzo_dwo = h2
    dcost_dwo = np.dot(dzo_dwo.T, dcost_dho * dho_dzo)

    # Second hidden layer
    dcost_dzo = dcost_dho * dho_dzo
    dzo_dh2 = wo
    dcost_dh2 = np.dot(dcost_dzo, dzo_dh2.T)
    dh2_dz2 = der_ReLu(z2)
    dz2_dw2 = h1
    dcost_dw2 = np.dot(dz2_dw2.T, dcost_dh2 * dh2_dz2)

    # First hidden layer
    dcost_dz2 = dcost_dh2 * dh2_dz2
    dz2_dh1 = w2
    dcost_dh1 = np.dot(dcost_dz2, dz2_dh1)
    dh1_dz1 = der_ReLu(z1)
    dz1_dw1 = inputs
    dcost_dw1 = np.dot(dz1_dw1.T, dcost_dh1 * dh1_dz1)

    # Derivatives for biases
    dcost_db3 = dcost_dzo
    dcost_db2 = dcost_dz2
    dcost_db1 = dcost_dh1 * dh1_dz1

   # Update weights
    w1 -=  learningRate * dcost_dw1
    w2 -=  learningRate * dcost_dw2
    wo -=  learningRate * dcost_dwo

    # Update biases
    b1 -= learningRate * dcost_db1
    b2 -= learningRate * dcost_db2
    b3 -= learningRate * dcost_db3
    
    
plt.plot(loss)   
plt.xlabel('Number of iterations')
plt.ylabel('Loss function for train data')
plt.suptitle('Loss function plot computed over the train set')
plt.show()


print(accuracy_score(outputs, ho.round()))
print(confusion_matrix(outputs, ho.round()))


#-------------------------------------------------------------------------#
# Validation set

loss2 = []
# Define new biases for the validation set
b1 = b1[0:len(inputs2)]
b2 = b2[0:len(inputs2)]
b3 = b3[0:len(inputs2)]

for i in range (800):
    
    # Forward propagation
    z1 = np.dot(x2, w1) + b1
    h1 = ReLu(z1)

    z2 = np.dot(h1, w2) + b2
    h2 = ReLu(z2)

    zo = np.dot(h2, wo) + b3 
    ho = sigmoid(zo) # output
    
    
    # Define cost function
    cost = ((1 / 2) * (np.power((ho - outputs2), 2)))
    loss2.append(cost[0])
    
    # Back-propagation    
    
    # Gradient for output layer
    dcost_dho = ho - outputs2
    dho_dzo = der_sigmoid(zo) 
    dzo_dwo = h2
    
    dcost_dwo = np.dot(dzo_dwo.T, dcost_dho * dho_dzo)

    # Gadient for second hidden layer
    dcost_dzo = dcost_dho * dho_dzo
    dzo_dh2 = wo
    dcost_dh2 = np.dot(dcost_dzo, dzo_dh2.T)
    dh2_dz2 = der_ReLu(z2)
    dz2_dw2 = h1
    dcost_dw2 = np.dot(dz2_dw2.T, dcost_dh2 * dh2_dz2)

    # Gradient for first hidden layer
    dcost_dz2 = dcost_dh2 * dh2_dz2
    dz2_dh1 = w2
    dcost_dh1 = np.dot(dcost_dz2, dz2_dh1)
    dh1_dz1 = der_ReLu(z1)
    dz1_dw1 = inputs2
    dcost_dw1 = np.dot(dz1_dw1.T, dcost_dh1 * dh1_dz1)

    dcost_db3 = dcost_dzo
    dcost_db2 = dcost_dz2
    dcost_db1 = dcost_dh1 * dh1_dz1

    # Update weights and biases
    w1 -=  learningRate * dcost_dw1
    w2 -=  learningRate * dcost_dw2
    wo -=  learningRate * dcost_dwo

    bv1 -=  learningRate * dcost_db1
    bv2 -=  learningRate * dcost_db2
    bv3 -=  learningRate * dcost_db3


plt.plot(loss2)   
plt.xlabel('Number of iterations')
plt.ylabel('Loss function for validate data')
plt.suptitle('Loss function plot computed over the validation set')
plt.show()

print(accuracy_score(outputs2, ho.round()))
#heatmap[rowno][colno] = accuracy_score(outputs2, ho.round())   ======= For Exercise 6 ===========
print(confusion_matrix(outputs2, ho.round()))