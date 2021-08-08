#Authors: DS291

# The functions below read in the dataset's gound truths then count the occurances of each target class. 

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import nibabel as nib
import os
from tensorflow.keras import backend as K
import pickle

# File path:
#path = "/home/ds291/Brain_Tumour_Segmentation_CNN_master/dataset/MICCAI_BraTS2020_TrainingData/"
path = './dataset_conv/X/'
label_path = './dataset_conv/Y/'

def read_image(file_path):
    img = nib.load(file_path)
    # the model to which this is being integrated reads the data in form (155, 240, 240), hence the transpose method
    img_data = img.get_fdata()
    return np.transpose(img_data, (2,1,0))

def read_data(path):

    my_dir = sorted(os.listdir(path))

    # While testing, limit dataset size. 
    limit = 369
    index = 0

    #training_names = []
    #ground_truth_names = []

    for p in tqdm(my_dir):

        #data = []
        gt = []

        if ('.csv' not in p) and (index < limit):

            index += 1

            data_list = sorted(os.listdir(path+p))

            # Ground truth images:
            seg = read_image(path + p + '/' + data_list[1])

            gt = np.asarray(seg, dtype = np.uint8)

            gt_name = 'y' + str(index) +'_gt.npy'

            np.save(gt_name, gt)

def load_data(path):

    my_dir = sorted(os.listdir(path))

    dataset = []

    #For each directory in the dataset, load in the corresponding segmentation, append it to the array.
    for f in tqdm(my_dir):
        print(f)
        data = np.load(path + f)
        print(data.shape)
        dataset.append(data)

    #Convert to a numpy array and return it. 
    dataset_conv = np.asarray(dataset)
    return dataset_conv

#Calculates class weitghts, saves them as a pkl file. 
def get_weights(dataset):

    print('dataset shape: ' + str(dataset.shape))

    flat_dataset = K.flatten(dataset)

    print('dataset shape: ' + str(flat_dataset.shape))
    
    #Count the occurances of each class label
    label_count = np.bincount(flat_dataset)

    #Calculates weights for each class as  (1 / number of occurances) * (2 * (dataset size))
    weight_0 = (1/label_count[0]) * (dataset.size*2)
    weight_1 = (1/label_count[1]) * (dataset.size*2)
  
    #Save class weights to a dictionary object
    weight_dictionary = {0:weight_0, 1:weight_1}

    print('Weight for class 0 (Background): ' + str(weight_0))
    print('Weight for class 1 (Tumour): '+ str(weight_1))

    #Save dictionary file to local directory for use by model. 
    dict_file = open("class_weights_bin.pkl", "wb")
    pickle.dump(weight_dictionary, dict_file)
    dict_file.close()

read_data(label_path)

data = load_data(label_path)

get_weights(data)