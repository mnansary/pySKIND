"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored


from coreLib.utils import data_input_fn,readh5

import numpy as np 
import matplotlib.pyplot as plt 


import tensorflow as tf 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense,Flatten

#--------------------------------------------------------------------------------------------------

tf.compat.v1.enable_eager_execution()
#--------------------------------------------------------------------------------------------------

class FLAGS:
    TFRECORDS_DIR  = '/home/ansary/RESEARCH/MEDICAL/Data/DataSet/TFRECORD/'
    IMAGE_DIM       = 64
    NB_CHANNELS     = 3
    BATCH_SIZE      = 1024
    SHUFFLE_BUFFER  = 2000
    MODE            = 'Train'
    
NB_TOTAL_DATA       = 26624 
NB_EVAL_DATA        = 1024
NB_TRAIN_DATA       = NB_TOTAL_DATA -  NB_EVAL_DATA 
CLASSES=['eczema','psoriasis']
#--------------------------------------------------------------------------------------------------
    
def check_data():
    dataset=data_input_fn(FLAGS,'NOT NEEDED')
    LEN=0
    for images,labels in dataset:
        print(images.shape)
        for i in range(images.shape[0]):
            print(labels[i])
            LEN+=1
            plt.imshow(np.squeeze(images[i]))
            plt.show()
            if LEN>10:
                break
        if LEN>10:
            break 
        
#--------------------------------------------------------------------------------------------------
N_EPOCHS            = 2
STEPS_PER_EPOCH     =  NB_TOTAL_DATA //FLAGS.BATCH_SIZE 
VALIDATION_STEPS    =  NB_EVAL_DATA //FLAGS.BATCH_SIZE 
#--------------------------------------------------------------------------------------------------

def train_in_fn():
    return data_input_fn(FLAGS,'Train')
def eval_in_fn():
    FLAGS.MODE='Eval'
    return data_input_fn(FLAGS,'Eval')
#--------------------------------------------------------------------------------------------------
def check_h5():
    X_H5='/home/ansary/RESEARCH/MEDICAL/Data/DataSet/Train/X_train.h5'
    Y_H5='/home/ansary/RESEARCH/MEDICAL/Data/DataSet/Train/Y_train.h5'
    X=readh5(X_H5)
    Y=readh5(Y_H5)
    for i in range(10):
        print(Y[i])
        plt.imshow(np.squeeze(X[i]))
        plt.show()
        


#--------------------------------------------------------------------------------------------------

def build():
    model = Sequential()
    model.add(Conv2D(64, (2, 2), input_shape=(64,64,3),activation='relu',padding='valid'))
    model.add(Flatten())
    model.add(Dense(2,activation = 'softmax'))
    return model

def tarin_debug():
    model = build()
    model.summary()
    model.compile(
        optimizer=tf.compat.v1.train.AdamOptimizer(),
        loss=tf.keras.losses.mean_absolute_error,
    )
    
    model.fit(
        train_in_fn(),
        epochs= N_EPOCHS,
        steps_per_epoch= STEPS_PER_EPOCH,
        validation_data=eval_in_fn(),
        validation_steps= VALIDATION_STEPS
    )
#--------------------------------------------------------------------------------------------------

check_h5()
check_data()
tarin_debug()
