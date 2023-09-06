from re import T
import stacked_mnist as stmn
import verification_net as vnet
from stacked_mnist import StackedMNISTData, DataMode
from autoencoder import autoencoder as ae
from variationalae import variationalae as vae
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from visualizator import *

os.environ['TF_CPP_MIN_LOG_LEVEL']= '3'
#### MAIN PROGRAM (ONE MODEL FOR EACH EXECUTION)

##Data that's going to be used
dataUsed = False
color = False
float = False
missing= False
modelDir = ""

##Verification Net that is going to be used
#verNet= None

## Encoder Class that's going to be used
autEncoderClass = None

#############################
#### AUXILIAR FUNCTIONS #####
#############################

def chooseDataModel():
    #print("MONO OR COLOR (0 - 1)")
    mc= "1" #input()
    #print("Binary or Float (0 - 1) ")
    bf= "0"  #input()
    #print("Complete or Missing (0 - 1)")
    cm = "0" #input()
    dMode= None
    global float
    global missing
    global color
    if (mc == "0" and bf == "0" and cm == "0"):
        dMode= DataMode.MONO_BINARY_COMPLETE

    elif (mc == "0" and bf == "1" and cm == "0"):
        float = True
        dMode= DataMode.MONO_FLOAT_COMPLETE

    elif (mc == "0" and bf == "0" and cm == 1):
        missing = True
        dMode= DataMode.MONO_BINARY_MISSING

    elif (mc == "0" and bf == "1" and cm == "1"):
        float = True
        missing = True
        dMode= DataMode.MONO_FLOAT_MISSING

    elif (mc == "1" and bf == "0" and cm == "0"):
        color = True
        dMode= DataMode.COLOR_BINARY_COMPLETE   

    elif (mc == "1" and bf == "1" and cm == "0"):
        color = True
        float = True
        dMode= DataMode.COLOR_FLOAT_COMPLETE

    elif (mc == "1" and bf == "0" and cm == 1):
        color = True
        missing = True
        dMode= DataMode.COLOR_BINARY_MISSING
    
    elif (mc == "1" and bf == "1" and cm == "1"):
        color = True
        missing = True
        float = True
        dMode= DataMode.COLOR_FLOAT_MISSING

    return StackedMNISTData(dMode)

def crossEntropy(targets:np.array, predictions:np.array, epsilon=1 - 12):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -(np.sum(targets * np.log(predictions + 1e-9))+np.sum((1-targets) * np.log(1-predictions + 1e-9))) / N
    return ce

def createModelsDir():
    aeDir= os.path.join(os.getcwd, "modelsAE")
    if(not os.path.exists(aeDir)):
        os.mkdir(aeDir)

    vaeDir = os.path.join(os.getcwd, "modelsVAE")
    if(not os.path.exists(vaeDir)):
        os.mkdir(vaeDir)


##################################
#### AUTO-ENCODER FUNCTIONS ######
##################################

def tarinAutoEnc():
    """
    trainAutoEnc:
    Returns Auto Encoder, Encoder, Decoder if the train has been completed
    """
    fit = autEncoderClass.fitAutoEncoder(dataUsed)
    if (not fit):
        print("Cannot execute Train AutoEncoder")
        exit()

def createAutoEncoder():
    if color:
        latentSize = 40
    else:
        latentSize = 80
    autEncoderClass.constructModel((28, 28, 3 if color else 1), latentSize, float)
    

#####PREDICTIONS OF THE AUTOENCODER --> WORKING (CHANGE THE NUMBER OF THE BATCH TO PREDICT) ####
def predictAutoEnc(generative: bool = False):
    global autEncoderClass
    if(not generative):
        dataBatch, targets = dataUsed.get_random_batch(False, 15)
        return autEncoderClass.predict(dataBatch), targets , dataBatch
    
    else:
        global color
        return autEncoderClass.generativeAutoEncoder(color)

########################################
###### VERIFICATION NET FUNCTIONS ######
######################################## 

def testDataset(predictions, targets, verNet):
    predict, acc = verNet.check_predictability(predictions, targets)
    print("RESULTS OF THE CHECK PREDICTABILITY FUNCTION")
    print("Predictability: ", predict)
    print("Accuracy: ", acc)

    coverage= verNet.check_class_coverage(predictions)
    print("RESULTS OF THE CHECK CLASS COVERAGE FUNCTION")
    print("Coverage: ", coverage)



####TRAIN VER MODEL WITH THE DATA USED###
def trainVerModel(type = 0):
    """
    trainVerModel 
    Actualizes de Verification Network variable wich will be used as
    the Verification Net to test the model
    """
    global color
    global dataUsed
    verNet= vnet.VerificationNet(file_name="modelAE" if type == 0 else "modelVAE")
    verNet.train(dataUsed, epochs=7)
    if(color):
        print("TRAINING WITH COLOR")
        verNet.train(dataUsed, 1, epochs=7)
        verNet.train(dataUsed, 2, epochs=7)
    allTrained = False
    while(not allTrained):
        rdBImages, rdBLables = dataUsed.get_random_batch(False, 100)
        _, acc = verNet.check_predictability(rdBImages, rdBLables)
        if(acc < 0.98):
            verNet.train(dataUsed, epochs=5)
            if(color):
                verNet.train(dataUsed, 1, epochs=5)
                verNet.train(dataUsed, 2, epochs=5)
        else: 
            allTrained = True
        
    return verNet




if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    visor = Visualizator()


    print("Use normal AutoEncoder or Variational (0 or 1)")
    aeType = input()

    ##AUTOENCODER
    if(aeType == "0"):
        print("Use a pre-created model or a new one (0 or 1)")
        load = input()
        
        ###PRECREATED MODEL
        if(load == "0"):
            print("Enter the name of the file where the model is saved")
            fileName = input()
            verNet = vnet(file_name= fileName)
        
        ###NEW MODEL
        else:
            print("Enter the name of the file where the model is going to be saved")
            fileName = input()
            verNet = vnet(file_name= fileName)
            

    
    ##VARIATIONAL

    else:
        print("Use a pre-created model or a new one (0 or 1)")
        pass
   


    


