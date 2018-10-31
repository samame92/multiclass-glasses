# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 15:51:53 2018

@author: ADMIIN
"""
"""IMPORTING"""
import keras
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import model_from_json
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.layers import Dropout
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import PIL
from PIL import Image
import scipy.io as sio
from keras.utils import to_categorical
import scipy.misc as mi
import matplotlib.pyplot as plt
import random
import cv2
import time
import io
import argparse
from googleAPIMethods import detect_web, detect_labels,localize_objects
from google.cloud import storage
from google.cloud import vision
from google.protobuf import json_format
"""END IMPORTING"""
"""PKL"""
import pickle
def savepickle(obj , filename):
    f = open(filename+'.pckl', 'wb')
    pickle.dump(obj, f)
    f.close()
def loadpickle(filename):
    f = open(filename+'.pckl', 'rb')
    obj = pickle.load(f)
    f.close()
    return obj
"""PKL"""

"""UTILITY"""
def preprocessInpuutImage(pathToImage, width , height):
    image = cv2.imread(pathToImage)### CHANGE PATH
    zimage = cv2.resize(image, (width, height))
    image = zimage.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    
    return image 
# import the necessary packages
import os

def list_images(basePath, contains=None):
    # return the set of files that are valid
    return list_files(basePath, validExts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"), contains=contains)

def list_files(basePath, validExts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"), contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename).replace(" ", "\\ ")
                yield imagePath
"""END"""
    
"""MAIN"""
#pathToImage = str(input('insert the path of the image to check'))  #input to the path image
pathToImage = 'NN/test_images/r1.jpg' #our image path
#str(input("INSERT THE URL OF THE IMAGE")) # input image url
imageURLfake = "https://www.alibaba.com" #fake
imageURLgood = "https://www.amazon.it" #good

#SAVING STUFF TO DON'T ABUSE ON THE API, IT COST $$$$$ 
annotations = detect_web(pathToImage) #The result of the web detection
labels = detect_labels(pathToImage) #the result of label detection
objects = localize_objects(pathToImage) # the result of object detection
savepickle(annotations, 'annotations')
savepickle(labels , 'labels')
savepickle(objects, 'obj')
#"""
"""LOADING STUFF
annotations = loadpickle('annotations')
labels = loadpickle('labels')
objects = loadpickle('obj')
 MODEL LOADING """
#The model are already trained, we just load the model to don't waste time, we'll show later them.
sunglassesModel = load_model('saved_model/sunglassesModel.model') #20Epochs
#raybanModel = load_model()
#multiclassModel = load_model()
image = preprocessInpuutImage(pathToImage , 128 , 128)
prediction  = sunglassesModel.predict(image)[0]
meanOfProbability = 0
if prediction.argmax() == 1:#if is sunglasses
    goOn = True
    raybanModel = load_model('saved_model/ourModel.model')
    #multiclassModel = load_model()
    meanOfProbability = (float(labels['sunglasses']) + float(objects['sunglasses']))/2
    if meanOfProbability <= 0.8:
        print("Google check said NOPE , no sunglasses in the picture, in google we trust.")
        quit()
        
else: #not sunglasses
    if ('sunglasses' in labels) and ('sunglasses' in objects):
        meanOfProbability = (float(labels['sunglasses']) + float(objects['sunglasses']))/2
        if meanOfProbability >= 0.8:
            ourPrediction = False
            goOn = True
            raybanModel = load_model('saved_model/ourModel.model')
            print("Not 100% sure, that could be a sunglasses with probability",meanOfProbability,", let's go on.")
        else:
            print("No sunglasses in the picture")
            quit()
                
    else:
        print("No sunglasses in the picture")
        quit()
       
#print((time.time() - t0)/1000 , 's')
rayBanPrediction = raybanModel.predict(image)[0]      
            
            
            
            
            
            
            