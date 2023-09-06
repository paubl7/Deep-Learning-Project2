from logging import raiseExceptions
import keras
from keras import layers, Model
import stacked_mnist as stkMn
import os
from tensorflow import keras
import tensorflow as tf
from keras.losses import binary_crossentropy
from keras import backend as K
from numpy import random as rnd
import numpy as np


class variationalae():

    def __init__(self):
        self.encoder = None
        self.decoder = None
        self.autoEncoder = None
        self.z = None
        self.zMean = None
        self.zLogVar = None
        self.modelCreated = False

    def constructModel(self, inputShape, latentSize, loadWeights: bool = False, fileName:str = ""):
        self.modelCreated = True 

        c = 2 if latentSize == 80 else 1
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
        ##DAFUQ IS HE DOIN
        self.zMean = layers.Dense(latentSize)(x)
        self.zLogVar = layers.Dense(latentSize)(x)
        self.z = layers.Lambda(self.sampling, output_shape=(latentSize,))([self.zMean, self.zLogVar])
        ####
        self.encoder = Model(inputEncoder, self.z, name="encoder")

        # Creating the decoder
        inputDecoder = layers.Input(shape=(latentSize,))
        x = layers.Dense(64 * c, activation="relu", use_bias=True)(inputDecoder)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Reshape((4, 4, 4*c))(x)
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
        self.autoEncoder = Model(inputEncoder, output, name='auto_encoder')

        self.autoEncoder.compile(optimizer='adam', loss=self.elboLoss)

        print("Model created succesfully")

    
    def elboLoss(self, true, pred):
        # Elbo loss =  binary cross-entropy + KL divergence
        recLoss = K.mean(binary_crossentropy(true, pred), axis=[1, 2])
        kLoss = -0.5 * K.mean(1 + self.zLogVar - K.square(self.zMean) - K.exp(self.zLogVar), axis=[-1])
        return 0.5 * recLoss + 0.5 * kLoss

    def sampling(self, args):
        zMean, zLogVar = args
        batch = K.shape(zMean)[0]
        dim = K.int_shape(zMean)[1]
        epsilon = K.random_normal(shape=(batch, dim), dtype=tf.float32)
        return zMean + K.exp(zLogVar) * epsilon

    