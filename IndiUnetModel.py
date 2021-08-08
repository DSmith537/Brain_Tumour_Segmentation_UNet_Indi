#Authors: DS291

# The functions below are adapted from the Github Repository https://github.com/carinanorre/Brain-Tumour-Segmentation-Dissertation, 
# accessed on 01/07/2021

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D, Conv2D, BatchNormalization, concatenate, Input, Dropout,Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from DiceScore import *
import matplotlib.pyplot as plt
import pickle
from DataGen import DataGenerator
import LossFunctions as losses

#This sets the Tensorflow random seed, increasing reproducibility of the model
tf.random.set_seed(1234)


########################################################################################################################################################
#
#                                                           U-NET MODEL ARCHITECTURE:
#
#
#                                       Input                                                           Output
#                                           block1                                              decode_block4
#                                               block2                                      decode_block3
#                                                   dropout1                            dropout2
#                                                       block3                      decode_block2
#                                                           block4             decode_block1
#                                                                   block_5
#
########################################################################################################################################################

#   Input Layer: just defines the shape of the input image
input_ = Input(shape=(128, 128,4), name='input')

##########################################################      ENCODING PATH       ####################################################################

# Encoding Block Architecture

#   Two convolutional layers - Conv2D(filter_size, kernel_size, activation, strides=(1, 1), padding='valid', dilation_rate=(1, 1),  
#                                       activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros')

#   Normalisation function - BatchNormalisation(axis=-1,momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer="zeros",
#                                               gamma_initializer="ones", moving_mean_initializer="zeros", moving_variance_initializer="ones",)

#   Pooling layer - MaxPooling2D(pool_size = (2,2), strides = pool_size, padding = 'valid', data_format = 'image_data_format' )


block1_conv1 = Conv2D(16, 3, padding='same', activation='relu', name='block1_conv1')(input_)
block1_conv2 = Conv2D(16, 3, padding='same', activation='relu', name='block1_conv2')(block1_conv1)
block1_norm = BatchNormalization(name='block1_batch_norm')(block1_conv2)
block1_pool = MaxPooling2D(name='block1_pool')(block1_norm)

block2_conv1 = Conv2D(32, 3, padding='same', activation='relu', name='block2_conv1')(block1_pool)
block2_conv2 = Conv2D(32, 3, padding='same', activation='relu', name='block2_conv2')(block2_conv1)
block2_norm = BatchNormalization(name='block2_batch_norm')(block2_conv2)
block2_pool = MaxPooling2D(name='block2_pool')(block2_norm)

block3_conv1 = Conv2D(64, 3, padding='same', activation='relu', name='block3_conv1')(block2_pool)
block3_conv2 = Conv2D(64, 3, padding='same', activation='relu', name='block3_conv2')(block3_conv1)
block3_norm = BatchNormalization(name='block3_batch_norm')(block3_conv2)
block3_pool = MaxPooling2D(name='block3_pool')(block3_norm)

block4_conv1 = Conv2D(128, 3, padding='same', activation='relu', name='block4_conv1')(block3_pool)
block4_conv2 = Conv2D(128, 3, padding='same', activation='relu', name='block4_conv2')(block4_conv1)
block4_norm = BatchNormalization(name='block4_batch_norm')(block4_conv2)
block4_pool = MaxPooling2D(name='block4_pool')(block4_norm)

#BOTTOM OF U
block5_conv1 = Conv2D(256, 3, padding='same', activation='relu', name='block5_conv1')(block4_pool)

##########################################################      DECODING PATH       ####################################################################

# Decoding Block Architecture

#   Transposed Convolution layer - Conv2DTranspose(filters, kernel_size, strides, padding, dilation_rate = (1,1), activation, kernel_initializer='glorot_uniform')

#   Concatenation layer - concatenate(prev_block_norm, prev_up_pool)

#   Convolution layer - Conv2D(filter_size, kernel_size, activation, strides=(1, 1), padding, dilation_rate=(1, 1),  
#                                       activation, kernel_initializer='glorot_uniform', bias_initializer='zeros')


up_pool1 = Conv2DTranspose(128, 3, strides=(2, 2), padding='same', activation='relu', name='up_pool1')(block5_conv1)
merged_block1 = concatenate([block4_norm, up_pool1], name='merged_block1')
decod_block1_conv1 = Conv2D(64, 3, padding='same', activation='relu', name='decod_block1_conv1')(merged_block1)

up_pool2 = Conv2DTranspose(64, 3, strides=(2, 2), padding='same', activation='relu', name='up_pool2')(decod_block1_conv1)
merged_block2 = concatenate([block3_norm, up_pool2], name='merged_block2')
decod_block2_conv1 = Conv2D(32, 3, padding='same', activation='relu', name='decod_block2_conv1')(merged_block2)

up_pool3 = Conv2DTranspose(32, 3, strides=(2, 2), padding='same', activation='relu', name='up_pool3')(decod_block2_conv1)
merged_block3 = concatenate([block2_norm, up_pool3], name='merged_block3')
decod_block3_conv1 = Conv2D(16, 3, padding='same', activation='relu', name='decod_block3_conv1')(merged_block3)

up_pool4 = Conv2DTranspose(16, 3, strides=(2, 2), padding='same', activation='relu', name='up_pool4')(decod_block3_conv1)
merged_block4 = concatenate([block1_norm, up_pool4], name='merged_block4')
decod_block4_conv1 = Conv2D(2, 3, padding='same', activation='relu', name='decod_block4_conv1')(merged_block4)


##########################################################      OUTPUT       ####################################################################

# Output Architecture
#   pre-output convolutional layer
#   output convolutional layer
#   Model declaration (unet_model = Model(inputs, outputs))

pre_output = Conv2D(2, 1, padding='same', activation='relu', name='pre_output')(decod_block4_conv1)
output = Conv2D(4, 1, padding='same', activation='softmax', name='output')(pre_output)

modelUNet = Model(inputs=input_, outputs=output)
print(modelUNet.summary())


##########################################################      EXECUTION        ###############################################################


image_path = './dataset_conv/X/'
label_path = './dataset_conv/Y/'
total_records = 369

#Lists with number of records
Xrecords = list(range(1,total_records))
Yrecords = list(range(1,total_records))

#Creating numpy arrays that contain the IDs of training/validation/test data
X_train, X_test, Y_train, Y_test = train_test_split(Xrecords, Yrecords, test_size=0.10, random_state=1234)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=1234)

train_idx = X_train
val_idx = X_val

#Saving test data for Evaluation
np.save("./dataset_conv/X_train", X_train)
np.save("./dataset_conv/Y_train", Y_train)
np.save("./dataset_conv/X_val", X_val)
np.save("./dataset_conv/Y_val", Y_val)
np.save("./dataset_conv/X_test", X_test)
np.save("./dataset_conv/Y_test", Y_test)

# Generator function call for loading bulk data in batches
training_generator = DataGenerator(train_idx, image_path, label_path)
validation_generator = DataGenerator(val_idx, image_path, label_path)

#Opening weights file to initialise the starting weights for dice loss function
weights_file = open("class_weights.pkl", "rb")
class_weights = pickle.load(weights_file)
weights_file.close
weights_list = list(class_weights.values())

print(weights_list)

#Process
# Compile model with model.compile(optimiser, loss, metrics)
# Apply early stopping with callbacks.EarlyStopping(patience, monitor)
# Fit the model with Model.fit(x,y,validation_data, batch_size, epochs, shuffle, callbacks)
# Save the model with model.Save(path, overwrite)

#Optimiser parameters
epochs = 60
learn_rate = 7*1e-4
decay_rate = learn_rate/epochs
momentum_rate = 0.8

#Optimisers:
#opt = tf.keras.optimizers.SGD(learning_rate=learn_rate, momentum=momentum_rate, decay = decay_rate)
#opt =  tf.keras.optimizers.RMSprop(learning_rate=learn_rate)
opt = Adam(learning_rate=learn_rate)

#Loss function selection
#chosen_loss = 'MSE'
#chosen_loss = 'MSE_weighted'
#chosen_loss = 'Dice'
#chosen_loss = 'Dice_weighted'
#chosen_loss = 'Cross_Entropy'
#chosen_loss = 'Cross_Entropy_weighted'
chosen_loss = 'Combo'
#chosen_loss = 'Combo_weighted'
#chosen_loss = 'Focal'

print("Chosen loss function: " + chosen_loss)

# The model is compiled with the dice_loss_function and the dice_function metric:
print("About to compile...")
modelUNet.compile(optimizer=opt, loss=losses.get_loss(chosen_loss, weights_list), metrics=[dice_function])

# EarlyStopping is applied incase the model stops improving with each epoch, increasing patience increases time before stopped:
callbacks = [tf.keras.callbacks.EarlyStopping(patience=4, monitor='val_loss')]

#Train the model
history = modelUNet.fit(
	x=training_generator,
	validation_data=validation_generator,
	epochs=epochs, callbacks=callbacks)

##########################################################      POST TRAINING        ###############################################################

modelUNet.save('./models/indi-UNet_'+ chosen_loss +'.h5', overwrite=True)
print("ModelSaved Successfully")

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['dice_function'])
plt.plot(history.history['val_dice_function'])
plt.title('Model Accuracy')
plt.ylabel('Dice_score')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./models/plots/IndiUnet_Accuracy_' + chosen_loss +'.png')

#Clears plot cache
plt.clf()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Dice Loss')
plt.ylabel('Dice_loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./models/plots/IndiUnet_Loss_' + chosen_loss +'.png')