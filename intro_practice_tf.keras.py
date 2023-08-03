# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 09:47:48 2023

@author: lauren
"""

import tensorflow as tf 
mnist=tf.keras.datasets.mnist 
(x_train, y_train), (x_test, y_test)= mnist.load_data()   #loads and processes the mnist data


x_train=tf.keras.utils.normalize(x_train, axis=1)         #normalizes the input data (to between 0 and 1)
x_test=tf.keras.utils.normalize(x_test, axis=1)


                                                          #build the model
model=tf.keras.models.Sequential()                            
model.add(tf.keras.layers.Flatten()) #flatten layer (insput layer) 
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))  #hidden layer, 128 nuerons in layer, rectified linear activation function
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))  #hidden layer, 128 nuerons in layer, rectified linear activation function
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax)) #number of classifcations, probablity distribution (use softmax)


 #setting more parameters for the training of the model
 #optimizer helps minimize the loss in the model 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'] ) #model trying to minimize loss
model.fit(x_train, y_train, epochs=3)       

#evaluate the model
val_loss, val_acc = model.evaluate(x_test, y_test)       

#val_loss=stores the validation loss obtained when evaluating the model on the test data. 
#provides a measure of how well the model performs during validation

#val_acc= holds the validation accuracy achieved by the model on the test data.
# It represents the percentage of correctly classified digits in the test dataset

#save and load model
model.save('num_reader.model')
new_model=tf.keras.models.load_model('num_reader.model')


#make predictions
predictions= new_model.predict([x_test])
print(predictions)



import numpy as np
print(np.argmax(predictions[19]))

import matplotlib.pyplot as plt
plt.imshow(x_train[19], cmap=plt.cm.binary)
plt.show()
#print(x_train[0])


plt.imshow(x_test[19]) 
plt.show
