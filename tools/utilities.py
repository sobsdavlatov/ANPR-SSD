import os 
import cv2
import csv
import json
import numpy as np 
import PIL as Image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

#Read csv and load data
def read_csv_and_load(csv_path, image_path):
    #Read csv file
    rows = open(csv_path).read().strip().split('\n')
    data = []
    targets = []
    filenames = []
    for row in rows:
        row = row.split(",")
        #Filename and bounding box 
        (filename, startX, startY, endX, endY) = row
        imagepaths = os.path.join(image_path, filename)
        image = cv2.imread(imagepaths)
        (h,w) = image.shape[:2]
        #Initialize starting point
        startX = float(startX) / w
        startY = float(startY) / h
        #Initialize ending point 
        endX = float(endX) / w
        endY = float(endY) / h
        #Lod iamge and define size 
        image = load_img(imagepaths, target_size = (224,224))
        image = img_to_array(image)
        #Appending to corresponing lists
        targets.append((startX, startY, endX, endY))
        filenames.append(filename)
        data.append(image)

    #Normalize data
    data = np.array(data, dtype="float32") / 255.0
    targets = np.array(targets, dtype="float32")
    X = data
    y = targets 

    return X, y 
