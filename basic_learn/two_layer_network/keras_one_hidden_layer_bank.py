#%%Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%Export Dataset from Excel or CSV
data=pd.read_csv('bank_dataset.csv')
X=data.iloc[:, 3:-1].values
Y=data.iloc[:,-1]

#%% Dealing with Missing Values
#Checking Missing Values First: https://chartio.com/resources/tutorials/how-to-check-if-any-value-is-nan-in-a-pandas-dataframe/
from sklearn.impute import SimpleImputer
if(pd.isna(X).any()):
    missing=1
    print("Missing Values Exists")
    imputer=SimpleImputer()
    X=imputer.fit_transform(X)
else:
    missing=0


#%% Dealing with Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer  
labelencoder1=LabelEncoder()
labelencoder2=LabelEncoder()
X[:,2]=labelencoder1.fit_transform(X[:,2])
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough') 
X = np.array(columnTransformer.fit_transform(X), dtype = np.str) 
# Remove First Column from Z to avoid Dummy Variable Trap: https://www.youtube.com/watch?v=BMNuPwUD5EA
# Dummy Variable Trap creates multi-colinearity
# Co-linearity Causes models to very sensitive to particular data : https://statisticsbyjim.com/regression/multicollinearity-in-regression-analysis/
X=X[:,1:]


#%% Test Training Splitting. It is required before Featue scaling
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

#%% Feature Scaling 
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

#%% Logistic Regression Model
#Check It can give 80% Accuracy on the Test Data
from sklearn.linear_model import LogisticRegression
classifier_linear=LogisticRegression(random_state=0)
classifier_linear.fit(X_train,Y_train)
Y_pred_linear=classifier_linear.predict(X_test)

#%% One layer Neural Network Model By Keras_Lib
import keras
from keras.models import Sequential
from keras.layers import Dense
#Keras Dense Layer Tutorial : https://www.tutorialspoint.com/keras/keras_dense_layer.htm
classifier_keras=Sequential()
#Input_Shape is only supported in the initial layer
classifier_keras.add(Dense(6, input_shape=(11,), activation='relu', kernel_initializer='uniform'))
#This is how a second layer should be added. But I have done with one layer only.
#classifier_keras.add(Dense(6,activation='relu',kernel_initializer='uniform'))
classifier_keras.add(Dense(1,activation='sigmoid',kernel_initializer='uniform'))

#Article for Optimizer : https://keras.io/api/optimizers/adam/
classifier_keras.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier_keras.fit(X_train, Y_train, batch_size=25, epochs=100)

Y_pred_keras=classifier_keras.predict(X_test)
Y_pred_keras=(Y_pred_keras>0.5)

#%% Evaluating the Model by Confusion matrix - Linear model vs One Layer Neural Network Model
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test, Y_pred_linear)
cm2=confusion_matrix(Y_test, Y_pred_keras)
percentage_accuracy_linear=((cm[0,0]+cm[1,1])/np.sum(cm))*100
percentage_accuracy_keras=((cm2[0,0]+cm2[1,1])/np.sum(cm2))*100