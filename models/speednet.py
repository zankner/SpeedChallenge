import tensorflow as tf
# Import layers:
from tensorflow.keras.layers import (
    Dense, Flatten, Conv2D, BatchNormalization, 
    concatenate, MaxPool2D
)
from tensorflow.keras import Model
from models.current import Current
from models.reference import Reference

#Defining network Below:
class SpeedNet(Model):
  def __init__(self):
    super(SpeedNet, self).__init__()
    # Define layers of the network:
    self.reference_layer = Reference()
    self.current_layer = Current()

    self.dense_0 = Dense(, activation='relu')
    self.dense_1 = Dense(, activation='relu')
    self.dense_2 = Dense(1)

    self.norm_0 = BatchNormalization()
    self.norm_1 = BatchNormalization()
    self.norm_2 = BatchNormalization()


  def call(self, ref_frame, cur_frame, training=False):
    # Call layers of network on input x
    # Use the training variable to handle adding layers such as Dropout
    # and Batch Norm only during training
    ref_vec = self.reference_layer(x)
    cur_vec = self.current_layer(x)
    x = concatenate(ref_vec, cur_vec)

    x = self.dense_0(x)
    x = self.norm_0(x)
    if training:
        x = Dropout(.1)(x)
    x = self.dense_1(x)
    x = self.norm_1(x)
    if training:
        x = Dropout(.1)(x)
    x = self.dense_2(x)
    return x
