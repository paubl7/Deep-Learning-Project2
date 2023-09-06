from logging import raiseExceptions
from keras import  Model
import stacked_mnist as stkMn
import os
from keras.losses import binary_crossentropy
from numpy import random as rnd
import numpy as np

######MODIFY THOSE FUNCTIONS
def saveModel(fileName: str, autoEncoder: Model):
    autoEncoder.save_weights(os.path.join(os.getcwd, "modelsAE", fileName))

def loadModel(fileName: str, autoEncoder: Model):
    """
    Loads the weights of a model created before
    """
    autoEncoder.load_weights(os.path.join(os.getcwd, "modelsAE", fileName))


def fitAutoEncoder(data: stkMn, autoEncoder: Model, color = False):
    """
    fitAutoEncoder: 
    Parameter: Data Generator (class)
    Fits the data choosen on the model only if it is already created. Returns True if the fit 
    process has been done without error and False if the model was not created before.
    """
    print("STARTING FITTING THE MODEL")
    if (autoEncoder.modelCreated):
        xTrain, _ = data.get_full_data_set() #data.get_random_batch(False, 64) 
        xVal, _ = data.get_random_batch(False, 128)
        if (not color):
            ###CHANGE 64/32 PARAMETERS TO HAVE A BETTER TRAINING
            autoEncoder.fit(xTrain, xTrain, validation_data=(xVal,xVal), epochs= 6)
            print("Model trained correctly")
            return True
        
        else:
            xTrainR = xTrain[:,:,:,[0]]
            xValR = xVal[:,:,:,[0]]
            xTrainG = xTrain[:,:,:,[1]]
            xValG = xVal[:,:,:,[1]]
            xTrainB = xTrain[:,:,:,[2]]
            xValB = xVal[:,:,:,[2]]

            autoEncoder.fit(xTrainR, xTrainR, validation_data=(xValR,xValR), epochs = 4)
            autoEncoder.fit(xTrainG, xTrainG, validation_data=(xValG,xValG), epochs = 4)
            autoEncoder.fit(xTrainB, xTrainB, validation_data=(xValB,xValB), epochs = 4)
            print("Model trained correctly")
    else:
        return False

    
def predict(self, data, autoEncoder:Model,  color = False):
    if(not color):
        return autoEncoder.predict(data)
        
    else:
        predR = autoEncoder.predict(data[:,:,:,[0]])
        predG = autoEncoder.predict(data[:,:,:[1]])
        predB = autoEncoder.predict(data[:,:,:,[2]])

            


def generativeAutoEncoder(autoEncoder: Model, color:bool = False):     
    data = []
    for i in range(0,20):
        if(color):
            data.append(rnd.randint(0,2,size=80))
        else:
            data.append(rnd.randint(0,2,size=40))
        
    for i in data:
        print(i)
    return autoEncoder.decoder.predict(np.array(data)), np.array(data)