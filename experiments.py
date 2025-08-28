import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100


N=[100,200,300]
M=[4,5,6]
data = {}

train_times = []
predict_times = []

for ele in N:
    for ele2 in M:
        X = pd.DataFrame(np.random.randint(0, 2, size=(ele, ele2)))
        y = pd.DataFrame( np.random.randint(0, 2, size=ele))
        tree = DecisionTree(criterion="information_gain",)
        startTime = time.time()
        tree.fit(X,y)
        train_times.append((ele,ele2, time.time() - startTime))
        
        startTime = time.time()
        tree.predict(X)
        predict_times.append((ele,ele2, time.time() - startTime))

data['Discrete i/p discrete o/p'] ={
    'i/p times' : train_times,
    'o/p times' : predict_times
}

train_times = []
predict_times = []
for ele in N:
    for ele2 in M:
        X = pd.DataFrame( np.random.randn(ele, ele2))
        y = pd.DataFrame(np.random.randint(0, 2, size=ele))
        tree = DecisionTree(criterion="information_gain",)
        startTime = time.time()
        tree.fit(X,y)
        train_times.append((ele,ele2, time.time() - startTime))
        
        startTime = time.time()
        tree.predict(X)
        predict_times.append((ele,ele2, time.time() - startTime))
        
data['Real i/p discrete o/p'] ={
    'i/p times' : train_times,
    'o/p times' : predict_times
}

train_times = []
predict_times = []
for ele in N:
    for ele2 in M:
        X = pd.DataFrame(np.random.randn(ele, ele2))
        y = pd.DataFrame(np.random.randn(ele))
        tree = DecisionTree(criterion="mse",)
        startTime = time.time()
        tree.fit(X,y)
        train_times.append((ele,ele2, time.time() - startTime))
        
        startTime = time.time()
        tree.predict(X)
        predict_times.append((ele,ele2, time.time() - startTime))

data['Real i/p real o/p'] ={
    'i/p times' : train_times,
    'o/p times' : predict_times
}

train_times = []
predict_times = []
for ele in N:
    for ele2 in M:
        X = pd.DataFrame(np.random.randint(0, 2, size=(ele, ele2))) 
        y = pd.DataFrame(np.random.randn(ele))
        tree = DecisionTree(criterion="mse",)
        startTime = time.time()
        tree.fit(X, y)
        train_times.append((ele,ele2, time.time() - startTime))
        
        startTime = time.time()
        tree.predict(X)
        predict_times.append((ele,ele2, time.time() - startTime))

data['Discrete i/p real o/p'] ={
    'i/p times' : train_times,
    'o/p times' : predict_times
}

print(data)
# Function to create fake data (take inspiration from usage.py)
# ...
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
# Function to plot the results
# ...
# Other functions
# ...
# Run the functions, Learn the DTs and Show the results/plots