#%%Import Libraries
import numpy as np
import pandas as pd

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
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

#%% Feature Scaling 
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

#%% Do Vectorised Training in One layer Neural Network
# Taken the initial wight concept from https://hackernoon.com/how-to-initialize-weights-in-a-neural-net-so-it-performs-well-3e9302d4490f
#I got Training Accuracy as 70% and Test Accuracy As 0% because it's a small dataset.  
# Good Blog: https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607

training_no=X_train.shape[0]
feature_no=X_train.shape[1]
alpha=0.03
epochs=1000
#W=np.random.rand(X_train.shape[1],1)*np.sqrt(2/feature_no)
W=np.zeros((X_train.shape[1],1))
b=0
for i in range(epochs):
    Z=np.dot(W.T,X_train.T)+b
    Y_pred=1/(1+np.exp(-Z))
    dz=Y_pred-y_train      
    db=(1/training_no)* np.sum(dz)
    dw=(1/training_no)* np.dot(dz,X_train) #Derivative of Cost function w.r.t weights
    W=W-alpha*dw.reshape(W.shape[0],W.shape[1])                           #Update Weight
    b=b-alpha*db                           #Update Bias
    J=1/training_no* np.sum(-(y_train*np.log(Y_pred)+(1-y_train)*np.log(1-Y_pred)))
    print("Cost Function",J)
    

Z_pred_test=np.dot(W.T,X_test.T)+b
Y_pred_test=1/(1+np.exp(-Z_pred_test))
y_pred_test=np.zeros((1,Y_pred_test.shape[1]))
y_pred_train=np.zeros((1,Y_pred.shape[1]))
for i in range(y_pred_test.shape[1]):
    if(Y_pred_test[0,i]>=0.5):
        y_pred_test[0,i]=1
    else:
        y_pred_test[0,i]=0
        
for i in range(Y_pred.shape[1]):
    if(Y_pred[0,i]>=0.5):
        y_pred_train[0,i]=1
    else:
        y_pred_train[0,i]=0

#%% Evaluating the Model by Percentage Accuracy Calculation
print("Training accuracy=",(1-np.mean(np.abs(y_train-y_pred_train))))
print("Test accuracy=",(1-np.mean(np.abs(y_test-y_pred_test))))
