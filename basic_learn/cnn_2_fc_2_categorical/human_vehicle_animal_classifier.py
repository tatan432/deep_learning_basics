#%%Import Libraries
import numpy as np
#import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

train_path = "./human_vehicle_animal/training_set/training_set"
test_path = "./human_vehicle_animal/test_set/test_set"

#%% Image Augmentation
train_datagen =  ImageDataGenerator(rescale=1./255,
                                    featurewise_center=True,
                                    featurewise_std_normalization=True,
                                    rotation_range=20,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    horizontal_flip=True)

test_datagen= ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_path,target_size=(50,50),batch_size=32,class_mode='categorical', color_mode='grayscale')
test_generator = test_datagen.flow_from_directory(test_path,target_size=(50,50),batch_size=32,class_mode='categorical', color_mode='grayscale')


print("Training Set Classes", train_generator.class_indices)
print("Test Set Classes",test_generator.class_indices)

#%% Deep Neural Network Model By Keras_Lib
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

#Keras Dense Layer Tutorial : https://www.tutorialspoint.com/keras/keras_dense_layer.htm

#%% Add Layers
classifier_keras=Sequential()
num_layers=4
conv_layer=2
fc_layers = num_layers-conv_layer

for i in range(conv_layer):  
    if (i==0):
        #Input_Shape is only supported in the initial layer
        classifier_keras.add(Conv2D(16, (3, 3), activation='relu',input_shape=(50, 50, 1))) 
        classifier_keras.add(MaxPooling2D((2, 2)))
    else:
        classifier_keras.add(Conv2D(32*i, (3, 3), activation='relu'))
        classifier_keras.add(MaxPooling2D((2, 2)))
        
classifier_keras.add(Flatten())

for i in range(fc_layers):  
    if (i==0):
        #Input_Shape is only supported in the initial layer
        classifier_keras.add(Dense(128, activation='relu', kernel_initializer='uniform'))
    elif (i==fc_layers-1):
         classifier_keras.add(Dense(4,activation='softmax',kernel_initializer='uniform'))

    else:
        classifier_keras.add(Dense(128,activation='relu',kernel_initializer='uniform'))
       

classifier_keras.summary()

#%% Train Classifier : For More output categorical_crossentropy
# Steps Per Epoch: ( Num_training_sample/batch_size)*2.5 . The 2.5 is multiplied as we are creating more images from Original Dataset by ImageDataGenerator
# Validation Steps: (NUm_test_sample/batch_size) - AS we are not creating any extra images by image transformation
# epochs : No of Epochs we want to run. 30 is a decent number

classifier_keras.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
history = classifier_keras.fit_generator(train_generator,
                              steps_per_epoch=300,
                              epochs= 30,
                              validation_steps=50,
                              validation_data=test_generator,
                             )
#history=classifier_keras.fit(X_train, Y_train, batch_size=25, epochs=1000)



acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


#%% Predicting a Single Sample
from keras.preprocessing import image
test_image = image.load_img('./human_and_non_human/single_set/dog_0565.jpg', target_size=(50,50), color_mode='grayscale')
test_image = image.img_to_array(test_image)
test_image = test_image.reshape(1,50,50,1)
Y_pred=classifier_keras.predict(test_image)





