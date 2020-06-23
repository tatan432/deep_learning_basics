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
Y_train=Y_train.values.reshape(len(Y_train),1) #Converting a series object to numpy array
Y_test=Y_test.values.reshape(len(Y_test),1) 

#%% Feature Scaling 
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

#%% Do Vectorised Training in two layer Neural Network : One hidden layer
# W1 and W2 wights at the layer 1 and 2. b1 and b2 are the corresponding biases
# A1 and A2 is the output of the layer 1 and 2
#Parameters for Training
training_no=X_train.shape[0]
feature_no=X_train.shape[1]
alpha=0.03
epochs=100
batch_size=25
neuron_num1=6 #First Hidden Layer Neuron Number
neuron_num2=1 #Output Layer Neuron Number


W1=np.random.rand(neuron_num1,feature_no)*np.sqrt(2/feature_no)
W2=np.random.rand(neuron_num2,neuron_num1)*np.sqrt(2/neuron_num1)
b1=np.zeros((neuron_num1,1))
b2=np.zeros((neuron_num2,1))
num_batch=int((X_train.shape[0])/batch_size)

for i in range(epochs):

    for j in range(num_batch):
        
        #Forward Propagation
        X_train_batch=X_train[j*batch_size:(j+1)*batch_size, :]
        Y_train_batch=Y_train[j*batch_size:(j+1)*batch_size, :]
        Z1=np.dot(W1,X_train_batch.T)+b1
        #Apply Relu activation in layer 1
        A1=np.maximum(Z1,0)
        Z2=np.dot(W2,A1)+b2
        A2=1/(1+np.exp(-Z2))
        
        #Backward Propagation
        
        #Second Layer        
        dZ2=(A2-Y_train_batch.T)      
        db2=(1/batch_size)* np.sum(dZ2)
        dW2=(1/batch_size)* np.dot(dZ2, A1.T) #Derivative of Cost function w.r.t weights
        W2=W2-alpha*dW2                       #Update Weight
        b2=b2-alpha*db2                       #Update Bias
        
        #First Layer
        drelu= (Z1>0)*1                       # Derivative of Relu
        dZ1=(np.dot(W2.T,dZ2)*drelu)          #dz1 is dA2/dz1 = dA2/d2*dz[2]/dA1*dA[1]/dZ[1]. Just match the dimension after derivative calculation.
        db1=(1/batch_size)*np.sum(dZ1, axis=1, keepdims=True)
        dW1=(1/batch_size)*np.dot(dZ1, X_train_batch)
        W1=W1-alpha*dW1                       #Update Weight
        b1=b1-alpha*db1                       #Update Bias

        if(j%10==0):
            loss=-(Y_train_batch.T *np.log(A2)+(1-Y_train_batch.T)*np.log(1-A2))
            J=(1/batch_size)* np.sum(loss)
            #print("Cost Function",J)
            print("epoch=",i,"batch=",j,"Training accuracy=",(1-np.mean(np.abs(Y_train_batch.T-A2))))
    

Z1_test=np.dot(W1,X_test.T)+b1
A1_test=np.maximum(Z1_test,0)
Z2_test=np.dot(W2,A1_test)+b2
A2_test=1/(1+np.exp(-Z2_test))
A2_test=(A2_test>0.5)*1


#%% Evaluating the Model by Percentage Accuracy Calculation

print("Test accuracy=",(1-np.mean(np.abs(Y_test-A2_test))))
