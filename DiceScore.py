#Authors: DS291

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import losses as Loss
import tensorflow as tf


# The DiceScore class contains functions pertaining to calculating the Dice coefficient for the models to use in training and validation.

# The functions below are adapted from the Github Repository https://github.com/carinanorre/Brain-Tumour-Segmentation-Dissertation, 
# accessed on 01/07/2021

###########################################################    Dice Functions    #######################################################################


##############################    For Model

def dice_function(y_ground, y_pred, smooth=1e5):
    #Main function used to compile the dice score
    y_ground_f = K.flatten(y_ground)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_ground_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_ground_f) + K.sum(y_pred_f) + smooth)
    return dice


def dice_loss_function (y_ground, y_pred):
    #Calculates loss function as 1-DiceScore, used to compile the models. 
    return 1 - dice_function(y_ground, y_pred)


#Calculates weighted dice function. Takes original loss function (dice_loss_function) and class weights as arguments
def weighted_dice(originalLossFunc, weightsList):

    def lossFunc(true, pred):

        axis = -1 #if channels last 

        #argmax returns the index of the element with the greatest value
        #done in the class axis, it returns the class index    
        classSelectors = K.argmax(true, axis=axis) 
        classSelectors = tf.cast(classSelectors, tf.int32)

        #considering weights are ordered by class, for each class true(1) if the class index is equal to the weight index   
        classSelectors = [K.equal(i, classSelectors) for i in range(len(weightsList))]

        #casting boolean to float for calculations each tensor in the list contains 1 where ground true class is equal to its index 
        classSelectors = [K.cast(x, K.floatx()) for x in classSelectors]

        #for each of the selections above, multiply their respective weight
        weights = [sel * w for sel,w in zip(classSelectors, weightsList)] 

        #sums all the selections, result is a tensor with the respective weight for each element in predictions
        weightMultiplier = weights[0]
        for i in range(1, len(weights)):
            weightMultiplier = weightMultiplier + weights[i]


        #calculate loss as dice_loss_function * weight_multiplier
        loss = originalLossFunc(true,pred) 
        loss = loss * weightMultiplier

        return loss
    return lossFunc

##############################    For Eval

def dice_tumour_regions(ground, prediction):
    # Calculates the dice score for the arrays that are no longer one-hot-encoded.
    if np.sum(ground) + np.sum(prediction) == 0:
        return 1
    else:
        return 2 * np.sum(ground * prediction) / (np.sum(ground) + np.sum(prediction))


#Multiclass
def dice_function_loop(ground_truth, preds):
    # This dice_function_loop prints each region's dice score using the above functions.
    tumour_part = ("Whole tumour", "Tumour core", "Enhancing tumour", "Background")
    tumour_part_functions = (get_whole_tumour, get_tumour_core, get_enhancing_tumour, get_background)
    rows = list()
    print(tumour_part)
    rows.append([dice_tumour_regions(func(ground_truth), func(preds)) for func in tumour_part_functions])
    print(rows)

#Binary
def dice_function_loop_binary(ground_truth, preds):
    # This dice_function_loop prints each region's dice score using the above functions.
    tumour_part = ( "Tumour", "Background")
    tumour_part_functions = (get_binary_tumour, get_background)
    rows = list()
    print(tumour_part)
    rows.append([dice_tumour_regions(func(ground_truth), func(preds)) for func in tumour_part_functions])
    print(rows)

##########################################################    Get Region Functions    ##################################################################

def get_whole_tumour(data):
    # The get_whole_tumour function isolates the pixels classed as 1,2,3, to create the whole tumour.
    return data > 0


def get_tumour_core(data):
    # The get_tumour_core function isolates the pixels classed as 1 and 3 to create the core tumour.
    return np.logical_or(data == 1, data == 3)


def get_enhancing_tumour(data):
    # The get_enhancing_tumour function isolates the pixels classed as 3 to create the enhancing tumour.
    return data == 3


def get_background(data):
    # The get_background function returns all the background pixels.
    return data == 0


def get_binary_tumour(data):
    # The get_background function returns all the background pixels.
    return data == 1
