import tensorflow as tf
# Import layers:
from tensorflow.keras.layers import (
    Dense, Flatten, Conv2D, BatchNormalization, 
    concatenate, MaxPool2D
)
from tensorflow.keras.layers import Layer

#Defining network Below:
class Current(Layer):
  def __init__(self):
    super(Current, self).__init__()
    # Define layers of the network:
    self.conv_0 = Conv2D(256, 2, activation='relu')
    self.conv_1 = Conv2D(128, 2, activation='relu')
    self.conv_2 = Conv2D(64, 2, activation='relu')

    self.pool_0 = MaxPool2D()

    self.norm_0 = BatchNormalization()
    self.norm_1 = BatchNormalization()
    self.norm_2 = BatchNormalization()

  def call(self, x, training=False):
    # Call layers of network on input x
    # Use the training variable to handle adding layers such as Dropout
    # and Batch Norm only during training
    x = self.conv_0(x)
    x = self.norm_0(x)
    if training:
        x = Dropout(.1)(x)
    x = self.conv_1(x)
    x = self.pool_0(x)
    x = self.norm_1(x)
    if training:
        x = Dropout(.1)(x)
    x = self.conv_2(x)
    x = self.norm_2(x)
    x = Flatten()(x)
    return x
