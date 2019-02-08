from tensorflow.python.layers.normalization import BatchNormalization as base_BatchNormalization

from .backend import learning_phase


class BatchNormalization(base_BatchNormalization):
    '''Override tf.layers.BatchNormalization with a version that uses
    learning_phase rather than requiring a training={True,False} when
    the layer is called.
    '''

    def __init__(self, *args, **kwargs):
        super(BatchNormalization, self).__init__(*args, **kwargs)
        #self._call_with_training = True
        self._uses_learning_phase = True

    def call(self, inputs, training='do-not-pass-arg', **kwargs):
        assert training == 'do-not-pass-arg', ('This BatchNormalization layer should be called '
                                               'without a training={True,False} arg, because it'
                                               ' sets training using tf_plus.learning_phase()')
        return super(BatchNormalization, self).call(inputs, training=learning_phase(), **kwargs)
