import os 
import cv2
import csv
import json
import numpy as np 
import PIL as Image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

#preprocess json into csv to simplify data loading

def preprocess_json(json_path, name):
    #read json file
    for filename in os.listdir(json_path):
        if filename.endswith(".json"):
            json_file = os.path.join(json_path, filename)
    with open(json_file, 'r') as f:
        data = json.load(f)

    file_names = []
    bounding_boxes = []
    for image in data['images']:
        file_names.append(image['file_name'])
    for annotation in data['annotations']:
        # Remove brackets from bounding boxes
        bbox = str(annotation['bbox']).strip('[]')
        # Split bounding box into separate values
        bbox_values = bbox.split(',')
        # Append each value to the list of bounding boxes
        for value in bbox_values:
            bounding_boxes.append(value)
    file_path = 'dataset/preprocessed/{}.csv'.format(name)
    # Write the filenames and bounding boxes to a CSV file
    exist = os.path.exists('dataset/preprocessed')
    if not exist:
        os.makedirs('dataset/preprocessed')
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Iterate over the data and write to file
        for i in range(len(file_names)):
            # Split bounding box values into separate variables
            x, y, w, h = bounding_boxes[i*4:(i+1)*4]
            # Write values to a row in the CSV file
            writer.writerow([file_names[i], x, y, w, h])

    return file_path

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