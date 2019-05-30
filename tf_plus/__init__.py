

from .backend import learning_phase
from .network import BaseLayer, Layers, SequentialNetwork
from .core import Lambda
from .normalization import BatchNormalization
from .util import setup_session_and_seeds, print_trainable_warnings
from .regularizers import l2reg
from .preprocessing import PreprocessingLayers
from .recurrent import BasicRNN, BasicLSTM
from .wrappers import DimDistributed, Distributed01, Distributed012, TimeDistributed

# Make some common stuff from TF available for easy import
import tensorflow as tf
Conv2D = tf.layers.Conv2D
MaxPooling2D = tf.layers.MaxPooling2D
Flatten = tf.layers.Flatten
Dense = tf.layers.Dense
he_normal = tf.keras.initializers.he_normal()    # Must call to prouduce initializer object
relu = tf.nn.relu
UpSampling2D = tf.keras.layers.UpSampling2D
