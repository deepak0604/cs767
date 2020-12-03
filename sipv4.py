# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:59:10 2020

@author: D107808
"""
# Building neural network with keras 
import os
import pandas as pd
import matplotlib.pyplot as plt
import calendar
import datetime
from keras.models import Sequential
from keras.layers import Dense
#from sklearn import preprocessing
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
# Convert class attribute- Revenue to integer
data.Revenue = data.Revenue.astype(int)
# Convert VisitorType attribute to integer
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

#Retrieve predictive variables and class variable - Assignment 3 code change
x = data

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

# make probability predictions with the model
predictions = model.predict(x_train)
# round predictions 
rounded = [round(x[0]) for x in predictions]
# summarize the first 5 cases
for i in range(5):
	print('%s => %d (expected %d)' % (x_train[i].tolist(), predictions[i], y_train[i]))

# plot feature importance
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from numpy import sort

# fit model no training data
modelXGB = XGBClassifier()
modelXGB.fit(x_train, y_train)
# plot feature importance
plot_importance(modelXGB)
plt.show()

# XGBoost Feature Importance Scores

# make predictions for test data and evaluate
y_pred = modelXGB.predict(x_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# Fit model using each importance as a threshold
thresholds = sort(modelXGB.feature_importances_)
for thresh in thresholds:
	# select features using threshold
	selection = SelectFromModel(modelXGB, threshold=thresh, prefit=True)
	select_X_train = selection.transform(x_train)
	# train model
	selection_model = XGBClassifier()
	selection_model.fit(select_X_train, y_train)
	# eval model
	select_X_test = selection.transform(x_test)
	y_pred = selection_model.predict(select_X_test)
	predictions = [round(value) for value in y_pred]
	accuracy = accuracy_score(y_test, predictions)
	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
