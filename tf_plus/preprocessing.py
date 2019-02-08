import tensorflow as tf
from .network import SequentialNetwork
from .core import Lambda


class PreprocessingLayers(SequentialNetwork):
    def __init__(self, shift_in=None):
        super(PreprocessingLayers, self).__init__()
        self.add(Lambda(lambda x: tf.to_float(x)))
        if shift_in is not None:
            self.add(Lambda(lambda x: x - shift_in))
