# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 18:16:02 2020

@author: Deepak Bhatia
"""
# Libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix

here=os.path.abspath(__file__)
input_dir=os.path.abspath(os.path.join(here ,os.pardir ))
fname='online_shoppers_intention.csv'
file = os.path.join(input_dir, fname)

abbr_to_num = {name: num for num, 
               name in enumerate(calendar.month_abbr) if num}

# Read file
data = pd.read_csv(file)

#Inspect and print data
# data.shape
print(data.shape)

print(data.head())

print(data.describe())

## Retrieve column names
print(data.columns)

print("-----SIP Data Pre-Processing-----")

# Check for nulls 
data.isnull().sum().sum()

# Check the Distribution of the data by class label - Revenue
plt.subplots(1)
sns.countplot(data['Revenue'], palette = 'Pastel1')
plt.title('Revenue', fontsize = 20)
plt.xlabel('Revenue?', fontsize = 15)
plt.ylabel('Count', fontsize = 10)

#Output to show the imbalance in Class distribution
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
# removing the target column revenue from x
x = x.drop(['Revenue'], axis = 1)

y = data['Revenue']

# checking the shapes
print("Shape of x:", x.shape)
print("Shape of y:", y.shape)


# Standarize the predictive attributes/variables
# scalar = StandardScaler()
# X = scalar.fit_transform(X)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.30, 
                                                    random_state=0)

print("Shape of x_train :", x_train.shape)
print("Shape of y_train :", y_train.shape)
print("Shape of x_test :", x_test.shape)
print("Shape of y_test :", y_test.shape)

x_train = x_train[['PageValues', 'ExitRates', 'BounceRates',
                   'ProductRelated_Duration', 'Administrative_Duration']]

x_test = x_test[['PageValues', 'ExitRates', 'BounceRates',
                 'ProductRelated_Duration', 'Administrative_Duration']]

# Note: I removed one of the highly correlated attributes of 
# 'ProductRelated_Duration' and 'ProductRelated’ from the top 
# six predictive attributes. 
scalar = StandardScaler()
X_Train = scalar.fit_transform(x_train)

X_Test = scalar.fit_transform(x_test)

# Build the models 
#Fit a Random Forest Classifier model
model = RandomForestClassifier()
model.fit(X_Train, y_train)
y_pred = model.predict(X_Test)

# Evaluate the Random Forest Classifier model
print("Training Accuracy for RandomForestClassifier:", model.score(X_Train, y_train))
print("Testing Accuracy for RandomForestClassifier:", model.score(X_Test, y_test))


# Print the performance metrics 
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

#Fit a Gradient Boosting model
model = GradientBoostingClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
#y_pred_prob = model.predict_proba(x_test)

# Evaluate the Gradient Boosting Classifier model
print("Training Accuracy for GradientBoostingClassifier:", 
      model.score(x_train, y_train))
print("Testing Accuracy for GradientBoostingClassifier:", 
      model.score(x_test, y_test))
# Print the probablities 
# print("probablity :", y_pred_prob) (Note: still working on it)
# Print the performance metrics 
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

# The following section is more to learn the application of SVM algorithm to build the ML models 
#Fit an SVM model - Linear Kernel
modelSVM = SVC(kernel='linear')
modelSVM.fit(x_train, y_train)

y_pred = modelSVM.predict(x_test)
# Evaluate the model
print("Training Accuracy SVM:", modelSVM.score(x_train, y_train))
print("Testing Accuracy SVM :", modelSVM.score(x_test, y_test))

# Print the performance metrics 
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

