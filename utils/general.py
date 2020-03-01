from keras.utils import plot_model
from IPython.display import Image
import random
import math
import string
import os

# Display Keras Model
def show_keras_model(model):
    filename = "".join(random.choices(string.ascii_uppercase, k=10)) + ".png"
    plot_model(model, to_file=filename, show_shapes=True, show_layer_names=True)
    image = Image(filename)
    os.remove(filename)
    return image

# sigmoid
def sigmoid(x):
    return 1.0 / (1 + math.exp(-x))

def tanh(x):
    return math.tanh(x)
