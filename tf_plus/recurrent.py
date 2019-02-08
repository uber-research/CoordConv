import tensorflow as tf
from tensorflow.contrib import rnn
from IPython import embed
from .network import Layers


class _RNNBase(Layers):
    '''Abstract base class for RNN models.
    As with Keras, expects input of shape (b, t, c).'''

    def call(self, inputs):
        unstacked_in = tf.unstack(tf.transpose(inputs, (1, 0, 2)))
        unstacked_outputs, states = rnn.static_rnn(self.cell, unstacked_in, dtype=inputs.dtype)
        output = tf.transpose(tf.stack(unstacked_outputs), (1, 0, 2))
        return output


class BasicRNN(_RNNBase):
    '''An RNN.'''

    def __init__(self, num_units, *args, **kwargs):
        super(BasicRNN, self).__init__(*args, **kwargs)
        self.num_units = num_units
        self.l('cell', rnn.BasicRNNCell(self.num_units))


class BasicLSTM(_RNNBase):
    '''An LSTM.'''

    def __init__(self, num_units, *args, **kwargs):
        super(BasicLSTM, self).__init__(*args, **kwargs)
        self.num_units = num_units
        self.l('cell', rnn.BasicLSTMCell(self.num_units))
