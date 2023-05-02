import os 
import csv
import json
import numpy as np 
import PIL as Image

#preprocess json into csv to simplify data loading

def preprocess_json(json_path, name):
    #read json file
    for filename in os.listdir(json_path):
        if filename.endswith(".json"):
            json_file = os.path.join(json_path, filename)
    with open(json_file, "r") as f:
        data = json.load(f)
    
    filenames = []
    bouninding_boxes = []
    for image in data["images"]:
        filenames.append(image["file_name"])
    for annotation in data["annotations"]:
        #Remove brackets from bounding boxes
        bbox = str(annotation["bbox"]).strip("[]")
        #Split bounding boxes into separate values
        bbox_values = bbox.split(",")
        #Appeding each value to the corresponding list
        for value in bbox_values:
            bbox_values.append(value)
    #Check whether the directory exist, if non-existant, create
    exist = os.path.exists("dataset/preprocessed/")
    if not exist:
        os.makedirs("dataset/preprocessed/")
    file_path = "dataset/preprocessed/{}.csv".format(name)
    #Write the filenames and bouding boxes to a CSV file
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        #Iterate over the data to write into a csv file
        for i in range(len(filenames)):
            #Split bouding box values into separate variables
            x, y, w, h = bouninding_boxes[i*4:(i+1)*4]
            #Write vaalues to a row in the CSV file
            writer.writerow([filenames[i], x, y,w,h])
    
    return file_path