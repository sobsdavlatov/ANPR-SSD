import os 
import cv2
import sys
import numpy as np
import csv
from sklearn.model_selection import train_test_split 
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

#Defining path to dataset and annotation files
dataset_path = "dataset/"
annots = os.path.join(dataset_path,"annots/_annotations.csv")
images_path = "dataset/images"

#Directory where preprocessed csv file will be stored
preprocessed_annots = os.path.join(dataset_path, "preprocessed/preprocessed_annots.csv")

#Columns from original csv file to be delted
columns_to_delete = [1,2,3]

with open(annots, 'r') as f_in, open(preprocessed_annots, 'w', newline='') as f_out:
    reader = csv.reader(f_in)
    writer = csv.writer(f_out)
    
    # Skip header row
    next(reader)
    
    for row in reader:
        # Delete specified columns
        row = [row[i] for i in range(len(row)) if i not in columns_to_delete]
        
        # Write updated row to output file
        writer.writerow(row)
#Getting data from preprocessed csv file
rows = open(preprocessed_annots).read().strip().split("\n")

data = []
targets = []
filenames = []

for row in rows:
    row = row.split(",")
    (filename, x1, y1, x2, y2) = row

    image_path = os.path.join(images_path, filename)
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]
    #scale the bounding box coordinates
    x1 = float(x1) / w
    y1 = float(y1) / h
    x2 = float(x2) / w
    y2 = float(y2) / h

    #Loading images and preprocessign them
    image = load_img(image_path, target_size = (224,224))
    image = img_to_array(image)

    data.append(image)
    targets.append((x1, y1, x2, y2))
    filenames.append(filename)
#Converting data to Numpy array and scaling the input pixel intensity
X = np.array(data, dtype="float32") / 255.0
y = np.array(targets, dtype="float32")


#Splitting data into training and validaton sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


