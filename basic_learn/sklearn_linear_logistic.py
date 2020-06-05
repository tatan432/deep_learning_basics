#%%Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%Export Dataset from Excel or CSV
data=pd.read_csv('small_data.csv')
X=data.iloc[:, :-1].values
Y=data.iloc[:,-1]

#%% Dealing with Missing Values
from sklearn.impute import SimpleImputer
imputer=SimpleImputer()
X[:,1:3]=imputer.fit_transform(X[:,1:3])


#%% Dealing with Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer  
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough') 
X = np.array(columnTransformer.fit_transform(X), dtype = np.str) 
labelencoder=LabelEncoder()
Y=labelencoder.fit_transform(Y)


#%% Test Training Splitting. It is required before Featue scaling
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#%% Feature Scaling 
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

#%% Logistic Regression Model
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

#%% Evaluating the Model by Confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
