#Authors: DS291

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import losses as Loss
import tensorflow as tf
from DiceScore import dice_function

#Returns selected loss function based on input string and class weights, for use by model and eval.
def get_loss(chosen_fn, class_weights):

    switcher = {
        'MSE' : Loss.mean_squared_error,
        'MSE_weighted' : weighted_loss(Loss.mean_squared_error, class_weights),
        'Dice' : dice_loss_function,
        'Dice_weighted' : weighted_loss(dice_loss_function, class_weights),
        'Cross_Entropy' : Loss.categorical_crossentropy,
        'Cross_Entropy_weighted' : weighted_loss(Loss.categorical_crossentropy, class_weights),
        'Combo' : dice_ce_loss,
        'Combo_weighted' : weighted_loss(dice_ce_loss, class_weights),
        'Focal' : categorical_focal_loss
    }
    loss_function = switcher.get(chosen_fn)

    return loss_function

# Implementation of Focal Loss (loss = -alpha*((1-p)^gamma)*log(p))
#    alpha -- the same as wighting factor in balanced cross entropy
#    gamma -- focusing parameter for modulating factor (1-p)
#
def categorical_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):

    # Define epsilon so that the backpropagation will not result in NaN
    # for 0 divisor case
    epsilon = K.epsilon()
    # Add the epsilon to prediction value
    #y_pred = y_pred + epsilon
    # Clip the prediction value
    y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
    # Calculate cross entropy
    cross_entropy = -y_true*K.log(y_pred)
    # Calculate weight that consists of  modulating factor and weighting factor
    weight = alpha * y_true * K.pow((1-y_pred), gamma)
        # Calculate focal loss
    loss = weight * cross_entropy
    # Sum the losses in mini_batch
    loss = K.sum(loss, axis=1)
    
    return loss
    

#Calculates combined cost function of dice and categorical cross entropy. 
def dice_ce_loss(y_ground, y_pred):
    alpha = 0.4
    beta = 1 - alpha
    dice_ce = (alpha * dice_function(y_ground, y_pred)) + (beta * Loss.categorical_crossentropy(y_ground, y_pred))
    return dice_ce

def dice_loss_function (y_ground, y_pred):
    #Calculates loss function as 1-DiceScore, used to compile the models. 
    return 1 - dice_function(y_ground, y_pred)


#Calculates weighted loss function. Takes original loss function and class weights as arguments
def weighted_loss(originalLossFunc, weightsList):

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