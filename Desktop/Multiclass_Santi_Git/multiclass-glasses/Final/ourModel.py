# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 11:42:52 2018

@author: FraudDetectionTeam
"""
import keras
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.layers import Dropout
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import PIL
from PIL import Image
import scipy.io as sio
from imutils import paths
from keras.utils import to_categorical
import scipy.misc as mi
import matplotlib.pyplot as plt

def createModel():
    model = Sequential()
    
    #Convolutional layers with 32 filters / kernels with a window size of 3×3
        
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(64,64,3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
     #Max pooling layer with window size 2×2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    #Conv layer with 64 filters
        
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    #Dense layer which performs the classification among 10 classes using a softmax layer
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
     
    return model