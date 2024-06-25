import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

''' (input: 784-dimensional vectors) ↧ 
[Dense (64 units, relu activation)] ↧ 
[Dense (64 units, relu activation)] ↧ 
[Dense (10 units, softmax activation)] ↧ 
(output: logits of a probability distribution over 10 classes) '''

inputs = keras.Input(shape=(784,))

# Just for demonstration purposes.
img_inputs = keras.Input(shape=(32, 32, 3))

dense = layers.Dense(64, activation="relu")
x = dense(inputs)

x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(10)(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")

model.summary()

keras.utils.plot_model(model, "my_first_model.png")
keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)