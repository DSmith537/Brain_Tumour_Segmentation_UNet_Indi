#Authors: DS291, AG360

# The functions below are adapted from the Github Repository https://github.com/carinanorre/Brain-Tumour-Segmentation-Dissertation, 
# accessed on 01/07/2021

import numpy as np
from tqdm import tqdm
import nibabel as nib
import os
from keras.utils.np_utils import to_categorical

#File path to dataset
path = '/home/ds291/Brain_Tumour_Segmentation_CNN_master/dataset/MICCAI_BraTS2020_TrainingData/'

#Reads individual image
def read_image(file_path):
    img = nib.load(file_path)
    # the model to which this is being integrated reads the data in form (90, 128, 128), hence the transpose method
    img_data = img.get_fdata()
    return np.transpose(img_data, (2,1,0))

# Normalizes the image
def normalize_image(img):  
    mean = img.mean()
    std = img.std()
    return (img - mean) / std

#Main function here that loads all of the separate modality images.
def load_data(path):

    my_dir = sorted(os.listdir(path))

    #In current dataset, there are 369 different groups of data - thus that is the limit, index is starting point
    #If set to 200 and index set to 100, it would read the 100 groups of images between these values.    
    limit = 369
    index = 0

    for p in tqdm(my_dir):
        gt = []
        x_image = np.zeros((4,155,240,240))
        if ('.csv' not in p) and (index < limit):

            index += 1

            data_list = sorted(os.listdir(path+p))

            # FLAIR images:
            img = read_image(path + p + '/' + data_list[0])
            x_image[0,:,:,:,] = normalize_image(img)

            # Ground truth images:
            img = read_image(path + p + '/' + data_list[1])
            seg = pre_process(img)

            # T1 images:
            img = read_image(path + p + '/'+ data_list[2])
            x_image[1,:,:,:,] = normalize_image(img)

            # T1ce (T1Gd) images:
            img = read_image(path + p + '/' + data_list[3])
            x_image[2,:,:,:,] = normalize_image(img)

            # T2 images:
            img = read_image(path + p + '/' + data_list[4])
            x_image[3,:,:,:,] = normalize_image(img)

            # Takes the middle slices of these images
            x_image_slice = x_image[:,30:120, 60:188, 60:188]
            x_image_slice = np.transpose(x_image_slice, (1,2,3,0))


            gt = np.asarray(seg, dtype = np.float32)

            #print(gt.shape)

            # Saving the final data sets to the current directory:
            img_name = './dataset_conv/X/x' + str(index) +'_img.npy'
            gt_name = './dataset_conv/Y/y' + str(index) +'_gt.npy'
            np.save(img_name, x_image_slice)
            np.save(gt_name, gt)


def pre_process(Y):
    Y = Y[30:120, 60:188, 60:188]
    Y = to_categorical(Y, num_classes=2)
    return Y

# Calling the load_data function using the path to the raw data, and saving the loaded images
# into data and ground truth arrays.
load_data(path)