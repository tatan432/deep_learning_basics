#Importing Dataset
train_path_humans = "./human_and_non_human/training_set/training_set/humans"
train_path_nhumans =  "./human_and_non_human/training_set/training_set/non-humans"
test_path_humans = "./human_and_non_human/test_set/test_set/humans"
test_path_nhumans = "./human_and_non_human/test_set/test_set/non-humans"
import os
from keras import preprocessing
import numpy as np

def load_image():
    path_humans = []
    for path in os.listdir(train_path_humans):
        if '.jpg' in path:
            path_humans.append(os.path.join(train_path_humans, path))
    path_nhumans = []
    for path in os.listdir(train_path_nhumans):
        if '.jpg' in path:
            path_nhumans.append(os.path.join(train_path_nhumans, path))
     
    path_humans_test = []
    for path in os.listdir(test_path_humans):
        if '.jpg' in path:
            path_humans_test.append(os.path.join(test_path_humans, path))    
            
    
    path_nhumans_test = []
    for path in os.listdir(test_path_nhumans):
        if '.jpg' in path:
            path_nhumans_test.append(os.path.join(test_path_nhumans, path))              
    
    
    tot_train_img=len(path_humans) + len(path_nhumans)
    tot_test_img =len(path_humans_test) + len(path_nhumans_test)
    
    train_img_x = np.zeros((tot_train_img, 50, 50, 1), dtype='float32')
    train_img_y = np.zeros((tot_train_img, 1), dtype='int32')
    
    test_img_x = np.zeros((tot_test_img, 50, 50, 1), dtype='float32')
    test_img_y = np.zeros((tot_test_img, 1), dtype='int32')
        
    
    #Storing the Images in the Training and Test Set Arrays
    
    for i in range(len(path_nhumans)):
        path = path_nhumans[i]
        img = preprocessing.image.load_img(path, color_mode= "grayscale", target_size=(50, 50))
        train_img_x[i] = preprocessing.image.img_to_array(img)
        train_img_y[i] = 0
    
    
    for i in range(len(path_humans)):
        path = path_humans[i]
        img = preprocessing.image.load_img(path,color_mode= "grayscale", target_size=(50, 50))
        train_img_x[i+len(path_nhumans)] = preprocessing.image.img_to_array(img)
        train_img_y[i+len(path_nhumans)] = 1
    
    
    for i in range(len(path_nhumans_test)):
        path = path_nhumans_test[i]
        img = preprocessing.image.load_img(path, color_mode= "grayscale",target_size=(50, 50))
        test_img_x[i] = preprocessing.image.img_to_array(img)
        test_img_y[i] = 0
    
    for i in range(len(path_humans_test)):
        path = path_humans_test[i]
        img = preprocessing.image.load_img(path, color_mode= "grayscale", target_size=(50, 50))
        test_img_x[i+len(path_nhumans_test)] = preprocessing.image.img_to_array(img)
        test_img_y[i+len(path_nhumans_test)] = 1       
      
    
    return train_img_x, train_img_y, test_img_x, test_img_y


