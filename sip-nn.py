# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 21:06:32 2020

@author: D107808
"""

# The following section is more to learn the application of Neural Nets algorithm to build the ML models 
# Using Keras to train and test the Neural Nets:
# Building neural network with keras 
import os
import datetime
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import calendar
abbr_to_num = {name: num for num, name in enumerate(calendar.month_abbr) if num}

here=os.path.abspath(__file__)
input_dir=os.path.abspath(os.path.join(here ,os.pardir ))
fname='online_shoppers_intention.csv'
file = os.path.join(input_dir, fname)

abbr_to_num = {name: num for num, 
               name in enumerate(calendar.month_abbr) if num}

# Read file
data = pd.read_csv(file)
# Convert Weekend attribute to integer
data.Weekend = data.Weekend.astype(int)

# Convert Revenue attribute to integer
data.Revenue = data.Revenue.astype(int)
# Convert visitorType attribute to integer
visitorType = {'Returning_Visitor': 0,'New_Visitor': 1, 'Other':2} 
data.VisitorType = [visitorType[item] for item in data.VisitorType] 

# Convert Months to integers
new_month = []

for i,v in data['Month'].items():
    frmt="%B" #if full month name is given
    if len(v) == 3:
        frmt="%b"
    datetime_object = datetime.datetime.strptime(v, frmt)
    month_number = datetime_object.month
    new_month.append(month_number)

      
data['MM'] = new_month
data['Month'] = data['MM']
data = data.drop(['MM'], axis=1)

#Retrieve predictor/dependent variables and class/independent variable
array = data.values
X = array[:,0:17]
Y = array[:,17]
# Standarize the predictor attributes
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X = scalar.fit_transform(X)

# define the keras model
model = Sequential()
model.add(Dense(34, input_dim=17, activation='relu'))
model.add(Dense(17, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, Y, epochs=150, batch_size=30)
# Split the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=0)
print("Shape of x_train :", x_train.shape)
print("Shape of y_train :", y_train.shape)
print("Shape of x_test :", x_test.shape)
print("Shape of y_test :", y_test.shape)

# evaluate the keras model (Full Data)
_, accuracy = model.evaluate(X, Y)
print('Accuracy: %.2f' % (accuracy*100))

# evaluate the keras model on training data
_, accuracy = model.evaluate(x_train, y_train)
print('Accuracy: %.2f' % (accuracy*100))

# evaluate the keras model on test data
_, accuracy = model.evaluate(x_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))

#Output

# make probability predictions with the model
predictions = model.predict(x_train)
# round predictions 
rounded = [round(x[0]) for x in predictions]
# summarize the first 5 cases
for i in range(5):
	print('%s => %d (expected %d)' % (x_train[i].tolist(), predictions[i], y_train[i]))

#Output
