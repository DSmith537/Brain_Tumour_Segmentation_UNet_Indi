#Authors: DS291

import numpy as np
#from DiceScore_bin import *
from DiceScore import dice_function, dice_function_loop
import LossFunctions as losses
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import pickle
from tensorflow.keras import backend as K
import tensorflow as tf
tf.random.set_seed(1234)

################################################################### Load Data Function ###############################################################
#Method loads in the validation or test data depending on binary argument test (tests if true).
def load_data(Data_path, Label_path, id_path, test):
    if test is False:
        #Load validation IDs.
        X = np.load(id_path + "X_val.npy")
        Y = np.load(id_path + "Y_val.npy")
    else:
        #Load test IDs.
        X = np.load(id_path + "X_test.npy")
        Y = np.load(id_path + "Y_test.npy")

    val_idx = X
    X = np.empty((len(val_idx),) + (90,128,128,4), dtype=np.float32) # this will be of the form (1, 4, 240, 240,155)
    Y = np.empty((len(val_idx),) + (90,128,128,4), dtype=np.float32) # this will be of the form (1, 240, 240,155)

    for i, ID in enumerate(val_idx):
        X[i,] = np.load(Data_path  + str(ID) + '_img.npy')
    X = X.reshape([-1,128,128,4])

    for i, ID in enumerate(val_idx):
        Y[i,] = np.load(Label_path + str(ID) + '_gt.npy')
    Y = Y.reshape([-1,128,128,4])

    return X, Y

################################################################### Output Image Functions ###############################################################
#Puts one image over another
def overlay(slice1, slice2, index, chosen_loss):
    #The ground truth slice should be the second slice
    plt.axis('off')
    plt.imshow(slice1, cmap=plt.cm.get_cmap('gray'))
    plt.savefig("brain.png", bbox_inches='tight')

    plt.axis('off')
    plt.imshow(slice2, cmap=plt.cm.get_cmap('gnuplot', 4))
    plt.savefig("seg.png", bbox_inches='tight')

    #After saving the images as .png, reopen as an image object to overlay 
    t1 = Image.open('brain.png')
    t2 = Image.open('seg.png')

    t1 = t1.convert("RGBA")
    t2 = t2.convert("RGBA")

    #Creates the blended image
    new_img = Image.blend(t1, t2, 0.5)
    new_img.save("./indi_overlays/" + chosen_loss + "/overlay" + str(index) + ".png","PNG")

def overlayKey(slice2):
    fig, ax = plt.subplots()
    plt.axis('off')
    cax = ax.imshow(slice2, cmap=plt.cm.get_cmap('gnuplot', 4))
    ax.set_title('Ground Truth')
    cbar = fig.colorbar(cax, ticks = [0.25,0.75,1.25,1.75], orientation = 'horizontal')
    cbar.ax.set_xticklabels(['Background','Necrotic Core', 'Edema', 'Enhancing Tumour'])
    plt.savefig("seg_key.png")

################################################################### Validation Data ###############################################################
Data_path = './dataset_conv/X/x'
Label_path = './dataset_conv/Y/y'
id_path = './dataset_conv/'

X_val, Y_val = load_data(Data_path, Label_path, id_path, False)
X_test, Y_test = load_data(Data_path, Label_path, id_path, True)

weights_file = open("class_weights.pkl", "rb")
weights = pickle.load(weights_file)
weights_file.close
weights_list = list(weights.values())

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

# Load in model:

unet_model = tf.keras.models.load_model('./models/indi-UNet_'+ chosen_loss +'.h5',
                                            custom_objects={'dice_ce_loss': losses.get_loss(chosen_loss, weights_list),
                                                            'dice_function': dice_function})

#True if we are running FINAL TESTS 
testing = True

################################################################### Validation Predictions ###############################################################
if testing is False:
    # Get predictions from validation data using model
    val_Y_pre = np.argmax(unet_model.predict(X_val), axis=-1)
    Y_val = np.argmax(Y_val, axis=-1)

    #Flatten the predictions and labels for the classification report.
    oneD_val_pre = K.flatten(val_Y_pre)
    oneD_Y = K.flatten(Y_val)

    classes = ['Background','Necrotic Core', 'Edema', 'Enhancing Tumour']
    report = classification_report(oneD_Y, oneD_val_pre, target_names = classes)

    # The prediction array and Y_val array are reshaped:
    val_Y_pre = val_Y_pre.reshape(-1, 128, 128, 1)
    Y_val_reshape = Y_val.reshape(-1, 128, 128, 1)

    # The dice_function_loop is called to evaluate the predictions with the ground truth labels:
    print("Dice scores using validation data: ", dice_function_loop(Y_val_reshape, val_Y_pre))
    print(report)

################################################################### Test Predictions ###############################################################

if testing is True:
    # Get predictions from validation data using model
    test_Y_pre = np.argmax(unet_model.predict(X_test), axis=-1)
    Y_test = np.argmax(Y_test, axis=-1)

    #Flatten the predictions and labels for the classification report.
    oneD_test_pre = K.flatten(test_Y_pre)
    oneD_Y_test = K.flatten(Y_test)

    classes = ['Background','Necrotic Core', 'Edema', 'Enhancing Tumour']
    test_report = classification_report(oneD_Y_test, oneD_test_pre, target_names = classes)

    # The prediction array and Y_val array are reshaped:
    test_Y_pre = test_Y_pre.reshape(-1, 128, 128, 1)
    Y_test_reshape = Y_test.reshape(-1, 128, 128, 1)

    # The dice_function_loop is called to evaluate the predictions with the ground truth labels:
    print("Dice scores using Test data: ", dice_function_loop(Y_test_reshape, test_Y_pre))
    print(test_report)

################################################################### Image saves ###############################################################
# The following loop takes slices 600,630 and saves their corresponding X data, Y data, and the predicted segmentation.
if testing is False:
    X_val = X_val.astype('uint8')
    val_Y_pre = val_Y_pre.astype('uint8')
    Y_val_reshape = Y_val_reshape.astype('uint8')

    for i in range(600,611):
        overlay(X_val[i, :, :, 0], val_Y_pre[i, :, :, 0], (str(i) + '_Prediction'), chosen_loss)
        overlay(X_val[i, :, :, 0], Y_val_reshape[i, :, :, 0], (str(i) + '_GroundTruth'), chosen_loss)

    overlayKey(val_Y_pre[600, :, :, 0])
else:
    X_test = X_test.astype('uint8')
    test_Y_pre = test_Y_pre.astype('uint8')
    Y_test_reshape = Y_test_reshape.astype('uint8')

    for i in range(780,790):
        overlay(X_test[i, :, :, 0], test_Y_pre[i, :, :, 0], (str(i) + '_Prediction'), chosen_loss)
        overlay(X_test[i, :, :, 0], Y_test_reshape[i, :, :, 0], (str(i) + '_GroundTruth'), chosen_loss)

    overlayKey(test_Y_pre[600, :, :, 0])