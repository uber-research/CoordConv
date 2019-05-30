import tensorflow as tf
from IPython import embed
from .network import Layers


class DimDistributed(Layers):
    '''A wrapper around other layers to reshape the given dims into a
    single leading dimension, apply a layer, and reshape back.

    Args:
        sdims: which dimensions to squash, or to distribute over.
            sdims = [0] is a no-op
            sdims = [0,1] is TimeDistributed
            sdims = [0,1,2] is like a doubly applied TimeDistributed
            sdims = [0,2] could be used, e.g., to apply an RNN expecting (b, t, c) input to a (b1, t, b2, c) input
        inner_layer: layer to apply to the flattened tensor
    '''

    def __init__(self, sdims, inner_layer, *args, **kwargs):
        super(DimDistributed, self).__init__(*args, **kwargs)
        assert isinstance(sdims, list) or isinstance(sdims, tuple)
        assert 0 in sdims, 'not distributing over the batch dimension is not supported'
        assert len(sdims) == len(set(sdims)), 'duplicate dim specified'
        self.sdims = sdims
        if self.sdims == list(range(self.sdims[-1]+1)):
            # sdims are something like [0], [0, 1], [0, 1, 2], ...
            self.transpose_required = False
        else:
            self.transpose_required = True
            #raise Exception('DimDistributed with transpose is not supported yet; add someday if needed')
        self.l('inner_layer', inner_layer)

    def call(self, inputs):
        n_sdims = len(self.sdims)
        
        list_in = isinstance(inputs, list)
        if not list_in:
            inputs = [inputs]

        if self.transpose_required:
            # Take shape of dims to be squashed from the first input. In the case of multiple
            # inputs, all must have the same leading shape.
            orig_sdim_shape = [tf.shape(inputs[0])[sd] for sd in self.sdims]
            flatter_inputs = []
            for inp in inputs:
                rank = len(inp.get_shape())
                # dims to keep non-squashed
                kdims = [ii for ii in range(rank) if ii not in self.sdims]

                transp_order = self.sdims + kdims
                inv_transp_order = [transp_order.index(ii) for ii in range(rank)]    # JBY: 85% sure this is right.

                inp_transpose = tf.transpose(inp, transp_order)
                # Could also replace -1 with prod
                flatter_shape = tf.stack([-1] + [tf.shape(inp)[kd] for kd in kdims], 0)
                flatter_inputs.append(tf.reshape(inp_transpose, flatter_shape))
            if list_in:
                flatter_out = self.inner_layer(flatter_inputs)
            else:
                flatter_out = self.inner_layer(flatter_inputs[0])
            
            rank = len(flatter_out.get_shape()) - 1 + n_sdims

            out_shape_pretransp = tf.stack(orig_sdim_shape + list(tf.shape(flatter_out)[1:]))
            out_pretransp = tf.reshape(flatter_out, out_shape_pretransp)
            
            # dims to keep non-squashed
            kdims = [ii for ii in range(rank) if ii not in self.sdims]
            transp_order = self.sdims + kdims
            inv_transp_order = [transp_order.index(ii) for ii in range(rank)]    # JBY: 85% sure this is right.

            out = tf.transpose(out_pretransp, inv_transp_order)

        else:
            # Take leading shape from first input. In the case of multiple
            # inputs, all must have the same leading shape.
            orig_leading_shape = tf.shape(inputs[0])[:n_sdims]
            flatter_inputs = []
            for inp in inputs:
                # Could also replace -1 with prod
                flatter_shape = tf.concat(([-1], tf.shape(inp)[n_sdims:]), 0)
                flatter_inputs.append(tf.reshape(inp, flatter_shape))
            if list_in:
                flatter_out = self.inner_layer(flatter_inputs)
            else:
                flatter_out = self.inner_layer(flatter_inputs[0])
            out_shape = tf.concat((orig_leading_shape, tf.shape(flatter_out)[1:]), 0)
            out = tf.reshape(flatter_out, out_shape)

        return out


class Distributed01(DimDistributed):
    '''A wrapper around other layers to reshape time (dim 1) into the batch dimension, apply a layer, and reshape back.'''

    def __init__(self, inner_layer, *args, **kwargs):
        super(Distributed01, self).__init__([0, 1], inner_layer, *args, **kwargs)


class Distributed012(DimDistributed):
    '''A wrapper around other layers to reshape time (dim 1) into the batch dimension, apply a layer, and reshape back.'''

    def __init__(self, inner_layer, *args, **kwargs):
        super(Distributed012, self).__init__([0, 1, 2], inner_layer, *args, **kwargs)


TimeDistributed = Distributed01
