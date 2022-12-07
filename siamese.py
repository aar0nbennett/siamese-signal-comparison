from pdb import set_trace

from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
import tensorflow as tf
import numpy as np

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class Siamese(tf.keras.Model):
    def __init__(self, chs, n_classes,
                 fs=128,
                 windows=1,
                 conv_kern_len=64, 
                 conv_filters=8, 
                 depthwise_filters=2, 
                 separable_filters=16,
                 separable_kern=16,
                 norm_rate=0.25,
                 drop_rate=0.5,  
                 drop_type='Dropout',
                 avg_kern_b1=4,
                 avg_kern_b2=8,
                 **kwargs):
        super(Siamese, self).__init__()
     
        drop_type = Dropout #self._select_drop_type(drop_type)
        
        # Block 1
        self.b1_conv2d = Conv2D(filters=conv_filters, 
                                kernel_size=(windows, conv_kern_len), 
                                padding='same',
                                use_bias=False,
                                data_format='channels_first',
                                name='b1_conv2d')
        self.b1_norm = BatchNormalization(axis=1, name='b1_batchnorm')
        self.b1_depth = DepthwiseConv2D(kernel_size=(chs, 1), 
                                        use_bias=False, 
                                        depth_multiplier=depthwise_filters,
                                        depthwise_constraint=max_norm(1.),
                                        data_format='channels_first',
                                        name='b1_depthwise')
        self.b1_norm2 = BatchNormalization(axis=1, name='b1_batchnorm_2')
        self.b1_activation = Activation('tanh', name='b1_activation')
        # Default = 128 // 4 = 32 Hz
        self.b1_pool = AveragePooling2D((1, avg_kern_b1),
                                        data_format='channels_first',
                                        name='b1_avg_pool')
        self.b1_dropout = drop_type(drop_rate, name='b1_dropout')
        
        # Block 2
        self.b2_separable = SeparableConv2D(filters=separable_filters, 
                                            kernel_size=(1, separable_kern),
                                            use_bias=False, 
                                            padding='same',
                                            data_format='channels_first',
                                            name='b2_separable')
        self.b2_norm = BatchNormalization(axis=1, name='b2_batchnorm')
        self.b2_activation = Activation('tanh', name='b2_activation')
        # Default = 128 // 4 // 8 = 4 Hz
        self.b2_pool = AveragePooling2D((1, avg_kern_b2),
                                        data_format='channels_first',
                                        name='b2_avg_pool')
        self.b2_dropout = drop_type(drop_rate, name='b2_dropout')
        
        # Block 3
        self.b3_flatten = Flatten(name='flatten')
        '''self.b3_dense  = Dense(units=n_classes, 
                               name='b3_dense', 
                               kernel_constraint=max_norm(norm_rate))'''

        # Output
        self.outputjr  = Dense(units=n_classes, activation='softmax', name='output')

    # Method used by fit method to run the model
    def call(self, x, training=False):
        # Split input into the pair inputs
        x_top = x[0]
        x_bottom = x[1]

        # Pass pairs through network
        top_vector = self.run_model(x_top, training)
        bottom_vector = self.run_model(x_bottom, training)
        
        # Combine feature vectors from top and bottom networks.
        distance = self.distance_measure(t=top_vector,b=bottom_vector,type='euc')

        # Apply final non-linear output layer
        return self.outputjr(distance)
            
    # completes similarity scoring between the two outputs and outputs the score
    def distance_measure(self, t, b, type):
        if type == 'euc':
            sum_square = tf.math.reduce_sum(tf.math.square(t - b), axis=1, keepdims=True)
            return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))
        elif type == 'man':
            return tf.math.reduce_sum(tf.math.abs(x - y), axis=1, keepdims=True)
        elif type == 'concat':
            return tf.keras.layers.concatenate([t, b])

    # passes the data through the model layers
    def run_model(self, x, training):
        # Block 1
        x = self.b1_conv2d(x)
        x = self.b1_norm(x, training=training)
        x = self.b1_depth(x)
        x = self.b1_norm2(x, training=training)
        x = self.b1_activation(x)
        x = self.b1_pool(x)
        x = self.b1_dropout(x, training=training)
        
        # Block 2
        x = self.b2_separable(x)
        x = self.b2_norm(x, training=training)
        x = self.b2_activation(x)
        x = self.b2_pool(x)
        x = self.b2_dropout(x, training=training)
        
        # Block 3
        x = self.b3_flatten(x)
        #x = self.b3_dense(x)

        return x