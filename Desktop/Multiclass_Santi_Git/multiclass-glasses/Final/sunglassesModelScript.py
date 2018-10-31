# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 11:07:45 2018

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
from alexNet import alexnet_model
from ourModel import createModel