import os 
import utilities as ut

#Define path to dataset 
base_dir = "dataset/"
train_dir = os.path.join(base_dir, "train")
valid_dir = os.path.join(base_dir, "valid")
test_dir = os.path.join(base_dir, "test")

#Preprocess json file into csv to simplify data loading
train_csv = ut.preprocess_json(train_dir, "train")
valid_csv = ut.preprocess_json(valid_dir, "valid")
test_csv = ut.preprocess_json(test_dir, "test")


#Load dataset into X and y
X_train, y_train = ut.read_csv_and_load(train_csv, train_dir)
X_valid, y_valid = ut.read_csv_and_load(valid_csv, valid_dir)
X_test, y_test = ut.read_csv_and_load(test_csv, test_dir)
