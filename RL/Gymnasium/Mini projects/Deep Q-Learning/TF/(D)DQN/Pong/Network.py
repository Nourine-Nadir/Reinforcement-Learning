from keras.layers import Activation, Dense, Conv2D, Flatten
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras import backend as K
from tensorflow import keras


class DQNetwork(keras.Model):
    def __init__(self,
                 lr,
                 n_actions,
                 input_dims,
                 fc1_dims):
        super().__init__()
        self.conv1 = Conv2D(filters=32,
                            kernel_size=8,
                            strides=(4),
                            activation='relu',
                            input_shape=(*input_dims,),
                            data_format='channels_first')  # channels_first: for 2D data: (channels, rows, cols)

        self.conv2 = Conv2D(filters=64,
                            kernel_size=4,
                            strides=2,
                            activation='relu',
                            data_format='channels_first')
        self.conv3 = Conv2D(filters=64,
                            kernel_size=3,
                            strides=1,
                            activation='relu',
                            data_format='channels_first')
        self.flatten = Flatten()
        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(n_actions)

    def call(self, inputs, training=False, **kwargs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)

        return self.fc2(x)
