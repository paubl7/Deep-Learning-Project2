from logging import raiseExceptions
import keras
from keras import layers, Model
import stacked_mnist as stkMn
import os
from tensorflow import keras
import numpy as np
from numpy import random as rnd
from keras import losses


class autoencoder:

    def __init__ (self):
        self.encoder = None
        self.decoder = None
        self.autoEncoder = None
        self.z = None
        self.zMean = None
        self.modelCreated = False

  

    ##CONSTRUCTION OF THE MODEL --> WORK
    def constructModel(self, inputShape, latentSize, binary, loadWeights: bool = False, fileName: str = ""):
        self.modelCreated = True 

        # Creating the encoder
        inputEncoder = keras.Input(shape=inputShape)
        x = layers.Conv2D(32, (3, 3), strides=2, activation="relu", use_bias=True, padding="same")(inputEncoder)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Conv2D(64, (3, 3), strides=2, activation="relu", use_bias=True, padding="same")(x)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Conv2D(64, (3, 3), strides=2, activation="relu", use_bias=True, padding="same")(x)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="relu", use_bias=True)(x)
        x = layers.BatchNormalization(axis=-1)(x)
        self.z = layers.Dense(latentSize, activation='linear', use_bias=True)(x)
        self.encoder = Model(inputEncoder, self.z, name="encoder")

        # Creating the decoder
        inputDecoder = layers.Input(shape=(latentSize,))
        x = layers.Dense(64 * 2, activation="relu", use_bias=True)(inputDecoder)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Reshape((4, 4, 4*2))(x)
        x = layers.Conv2DTranspose(32, (4, 4), activation="relu", use_bias=True,
                            padding='valid')(x)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Conv2DTranspose(64, (3, 3), activation="relu", strides=2, use_bias=True,
                            padding='same')(x)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Conv2DTranspose(inputShape[-1], (3, 3), strides=2, activation='sigmoid',
                            use_bias=True, padding='same')(x)
        self.decoder = Model(inputDecoder, x, name='decoder')

        output = self.decoder(self.encoder(inputEncoder))
        self.autoEncoder = Model(inputEncoder, output, name='autoEncoder')

        if binary:
            self.autoEncoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        else:
            self.autoEncoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        print("Model created succesfully")
        
        if(loadWeights):
            self.loadModel(fileName)
            print("Model loaded succesfully")
            

    ##TESTEJAR##
    def anomalyDetection(self, dataUsed: stkMn):
        ogImages, _= dataUsed.get_random_bacth(False, 100)
        pred = self.predict(ogImages)
        mse = losses.MeanSquaredError()
        return mse(ogImages, pred).numpy()
