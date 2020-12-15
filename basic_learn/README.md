
# Basic Learning:

In this section you will get to learn the basics of deep learning concpepts
Logistic Regression with small data set (10 training example):
This part partains to the basics of the machine learning. There are three folders:
1. One Layer Netowrk
2. Two Layer Network
3. Multi Layer Network

## One Layer Network:
One layer network can be thought as a simple linear regression model. So, a linear regression with Scikit Learn and one layer Neural Network have been implemented with Numpy.
1.	skleran_linear_logistic.py- It uses scikit-learn's linear regression model to fit the training data. It uses small_data.csv dataset. It is jus to understand how to preprocess the tabular data and perform logistic regression.
2.	numpy_one_layer_logistic.py- Logistic Regression can be thought of a one layer neural network.
So, a single layer neural network has been implemented with corresponding mathematical model. The mathematical model can be found in the video lecture by Andrew Ng. Just see these two videos -
A. https://www.youtube.com/watch?v=KKfZLXcF-aE&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=16
B. https://www.youtube.com/watch?v=2BkqApHKwn0&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=20

### Key Takeaway:
With these two tutorial you would get the basic knowlege of logistic regression, templates as how to pre-process dataset (The preprocessing step has been taken from Deep-Learning A-Z course material) and the need for larger dataset.

## Two Layer Network: 
A larger Bank dataset thereafter is taken to make a more powerful model. In this tutorial, we have used one layer network. There are two files:
1. numpy_one_hidden_layer_bank.py - As the name suggests, it is been implemented with Numpy.
2. keras_one_hidden_layer_bank.py - It is been implemented with Keras. 

### Key Takeaways: 
You would learn step by step how to build one hidden layer Neural Network and it will give you a foundation how subsequent layers can be added. 

## Multi Layer Deep Network:
A cat vs Non-Cat Picture has been taken a dataset in this Tutorial Section. This tutorial is very important as we will be building multi layer deep network. There will be several concepts will be explained by this tutorial. Such as Regularization, K-fold Cross Validation, Bias-Variance Tradeoff etc. This tutorial is under Progress. Agian there will be two files:
1. cat_noncat_classifier.py - This file is based on Keras Model. The number of layers and numer of hidden units are programmable.
2. dataset_load.py - It loads the dataset based from the h5 file.
