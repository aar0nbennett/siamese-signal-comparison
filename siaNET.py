from pdb import set_trace

import numpy as np
import tensorflow as tf

from deepbci.models.networks.eegnet import EEGNet

class Siamese(tf.keras.Model):
    def __init__(self, **eeg_kwargs):
        super(Siamese, self).__init__()
        # We do the [:-2] to remove the last 2 layers which correspond to the "head", used to output the number of classes. We are going to add our own head.
        self.encoder = tf.keras.Sequential(
            EEGNet(**eeg_kwargs).layers[:-2]
        )
        # Here we make the head (you might not need this), can remove activation if needed. It is set to 2 to output same or not.
        # You can also use sigmoid and 1 output.
        self.head = tf.keras.Sequential([tf.keras.layers.Dense(1, activation="sigmoid")])

    def call(self, x):
        x1, x2= x[:,0], x[:,1] 
        
        # x1 feature encoding
        x1_embed = self.encoder(x1)
        # x2 feature encoding
        x2_embed = self.encoder(x2)

        # add code below to check if x1_embed and x2_embed are the same. Use the head, l2, or l1 to compute this.
        # Combine feature vectors from top and bottom networks.
        distance = self.distance_measure(t=x1_embed,b=x2_embed,type='euc')
        print(distance.shape)

        # Apply final non-linear output layer
        output = self.head(distance)
        print(output.shape)
        return output
    
    def distance_measure(self, t, b, type):
        if type == 'euc':
            sum_square = tf.math.reduce_sum(tf.math.square(t - b), axis=1, keepdims=True)
            return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))
        elif type == 'man':
            return tf.math.reduce_sum(tf.math.abs(x - y), axis=1, keepdims=True)
        elif type == 'concat':
            return tf.keras.layers.concatenate([t, b])