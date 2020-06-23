#%%Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataset_load as dl

#%%Export Image Dataset from h5 file
# Dataset is the same as the used in the Andrew Ng's Deep Learning Lecture Series
# There are two different files for test and train. So splitting is not needed to be worried about

train_img_x, train_img_y, test_img_x, test_img_y=dl.load_image()


#%% Flatten the image
train_img_x=train_img_x.reshape(train_img_x.shape[0],-1) #Shape 0 has no of training examples. -1 flattens all other dimension.
test_img_x=test_img_x.reshape(test_img_x.shape[0],-1)
Y_train=train_img_y.reshape(len(train_img_y),1)
Y_test=test_img_y.reshape(len(test_img_y),1)

#%% Feature Scaling 

X_train=train_img_x/255;
X_test=test_img_x/255;



#%% Deep Neural Network Model By Keras_Lib
import keras
from keras.models import Sequential
from keras.layers import Dense
#Keras Dense Layer Tutorial : https://www.tutorialspoint.com/keras/keras_dense_layer.htm
classifier_keras=Sequential()
num_layers=5
hidden_unit=5

for i in range(num_layers):  
    if (i==0):
        #Input_Shape is only supported in the initial layer
        classifier_keras.add(Dense(hidden_unit, input_shape=(12288,), activation='relu', kernel_initializer='uniform'))
    elif (i==num_layers-1):
         classifier_keras.add(Dense(1,activation='sigmoid',kernel_initializer='uniform'))

    else:
        classifier_keras.add(Dense(hidden_unit,activation='relu',kernel_initializer='uniform'))
       

classifier_keras.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
history=classifier_keras.fit(X_train, Y_train, batch_size=25, epochs=1000)

plt.plot(classifier_keras.history.history['loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
Y_pred_keras=classifier_keras.predict(X_test)
Y_pred_keras=(Y_pred_keras>0.5)

#%% Evaluating the Model by Confusion matrix - Linear model vs One Layer Neural Network Model
from sklearn.metrics import confusion_matrix
cm2=confusion_matrix(Y_test, Y_pred_keras)
percentage_accuracy_keras=((cm2[0,0]+cm2[1,1])/np.sum(cm2))*100
