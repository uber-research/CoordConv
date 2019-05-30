
# Copyright (c) 2018 Uber Technologies, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import numpy as np
from orderedset import OrderedSet
import pdb
from IPython import embed

import tensorflow as tf



########################
# Reshaping helpers
########################

def tf_shape_notlast(tensor, append_neg1=False, name=None):
    '''Gets shape of tensor but not including the last dimension. E.g. for shape
    (a, b, c, ..., y, z)
    returns
    (a, b, c, ..., y)

    If append_neg1, returns
    (a, b, c, ..., y, -1)
    
    >>> inp = tf.placeholder('float32')
    >>> tsn = tf_shape_notlast(inp)
    >>> sess.run(tsn, {inp: np.zeros((2,3,4,5))})
    array([2, 3, 4], dtype=int32)
    >>> sess.run(tsn, {inp: np.zeros((2,3))})
    array([2], dtype=int32)
    >>> sess.run(tsn, {inp: np.zeros((2,))})
    array([], dtype=int32)
    >>> sess.run(tsn, {inp: 0})
    array([], dtype=int32)

    >>> tsn = tf_shape_notlast(inp, append_neg1=True)
    >>> sess.run(tsn, {inp: np.zeros((2,3,4,5))})
    array([2, 3, 4, -1], dtype=int32)
    >>> sess.run(tsn, {inp: np.zeros((2,3))})
    array([2, -1], dtype=int32)
    >>> sess.run(tsn, {inp: np.zeros((2,))})
    array([-1], dtype=int32)
    >>> sess.run(tsn, {inp: 0})
    array([-1], dtype=int32)    
    '''

    if append_neg1:
        return tf.concat(0, (tf.slice(tf.shape(tensor), [0], [tf.rank(tensor) - 1]), tf.expand_dims(-1, 0)), name=name)
    else:
        return tf.slice(tf.shape(tensor), [0], [tf.rank(tensor) - 1], name=name)

    
def tf_shape_last(tensor, prepend_neg1=False, squeeze=False, name=None):
    '''Gets length of last dimension of tensor. E.g. for shape
    (a, b, c, ..., y, z)
    returns
    (z)

    If prepend_neg1, returns
    (-1, z)

    If squeeze is True and prepend_neg1 is False, returns a scalar instead of 1D tensor.
    It is an error for prepend_neg1 and squeeze to both be true

    >>> inp = tf.placeholder('float32')
    >>> tsl = tf_shape_last(inp)
    >>> sess.run(tsl, {inp: np.zeros((2,3,4,5))})
    array([5], dtype=int32)
    >>> sess.run(tsl, {inp: np.zeros((2,3))})
    array([3], dtype=int32)
    >>> sess.run(tsl, {inp: np.zeros((2))})
    array([2], dtype=int32)
    >>> sess.run(tsl, {inp: 0})
    # Exception raised

    >>> tsl = tf_shape_last(inp, prepend_neg1=True)
    >>> sess.run(tsl, {inp: np.zeros((2,3,4,5))})
    array([-1, 5], dtype=int32)
    >>> sess.run(tsl, {inp: np.zeros((2,3))})
    array([-1, 3], dtype=int32)
    >>> sess.run(tsl, {inp: np.zeros((2))})
    array([-1, 2], dtype=int32)
    >>> sess.run(tsl, {inp: 0})
    # Exception raised    
    '''

    assert not (prepend_neg1 and squeeze), 'prepend_neg1 and squeeze are incompatible'

    if prepend_neg1:
        return tf.concat(0, (tf.expand_dims(-1, 0), tf.slice(tf.shape(tensor), [tf.rank(tensor) - 1], [1])), name=name)
    else:
        ret = tf.slice(tf.shape(tensor), [tf.rank(tensor) - 1], [1], name=name)
        if squeeze:
            return ret[0]
        else:
            return ret


def tf_batch_plus_shape(batch_ref_tensor, rest_of_shape, name=None):
    '''Creates a shape tensor containing the batch size of the given reference tensor plus specified trailing dimensions. E.g. for reference tensor shape
    (a, b, c, ...)
    and rest_of_shape
    (x, y, z, ...)
    returns
    (a, x, y, z, ...)
    '''

    if not (isinstance(rest_of_shape, tuple) or isinstance(rest_of_shape, tuple) or isinstance(rest_of_shape, tf.Tensor)):
        rest_of_shape = (rest_of_shape,)

    return tf.concat(0, (tf.slice(tf.shape(batch_ref_tensor), [0], [1]), rest_of_shape), name=name)


def tf_shape_first(batch_ref_tensor, squeeze=False, name=None):
    '''Creates a shape tensor containing the batch size of the given reference. E.g. for reference tensor shape
    (a, b, c, ...)
    returns
    (a,)
    '''

    first = tf.slice(tf.shape(batch_ref_tensor), [0], [1], name=name)
    if squeeze:
        return tf.reshape(first, [])
    else:
        return first


def tf_flatten(tensor, name=None):
    '''Flattens to rank 1 tensor'''
    return tf.reshape(tensor, [-1], name=name)

    
def tf_flatten_notlast(tensor, name=None):
    '''Flattens to rank 2 tensor, converting shapes like
    (a, b, c, ..., y, z)
    to
    (a*b*c*...*y, z)

    >>> inp = tf.placeholder('float32')
    >>> tfn = tf_flatten_notlast(inp)
    >>> sess.run(tfn, {inp: np.zeros((2,3,4,5))}).shape
    (24, 5)
    >>> sess.run(tfn, {inp: np.zeros((2,3))}).shape
    (2, 3)
    >>> sess.run(tfn, {inp: np.zeros((2))}).shape
    (1, 2)
    >>> sess.run(tfn, {inp: 0}).shape
    # Exception raised
    '''

    return tf.reshape(tensor, tf_shape_last(tensor, prepend_neg1=True), name=name)

    
def tf_reshape_like(tensor, pattern, name=None):
    '''Reshapes tensor to the shape of pattern (must have the same number of elements)

    >>> inp = tf.placeholder('float32')
    >>> pat = tf.placeholder('float32')
    >>> trl = tf_reshape_like(inp, pat)
    >>> sess.run(trl, {inp: np.zeros((2,3,4,5)), pat: np.zeros((6,20))}).shape
    (6, 20)
    >>> sess.run(trl, {inp: np.zeros((2,3,4,5)), pat: np.zeros((20,6))}).shape
    (20, 6)
    >>> sess.run(trl, {inp: np.zeros((15)), pat: np.zeros((3,5))}).shape
    (3, 5)
    '''
    
    return tf.reshape(tensor, tf.shape(pattern), name=name)


def tf_reshape_like_notlast(tensor, pattern, squeeze_last=False, name=None):
    '''Reshapes tensor to the shape of pattern for the first N-1
    dimensions and -1 for the last dimension.

    If squeeze_last, assume last dimension is length 1 and squeeze it
    (error if larger than 1).

    >>> inp = tf.placeholder('float32')
    >>> pat = tf.placeholder('float32')
    
    >>> trl = tf_reshape_like_notlast(inp, pat)
    >>> sess.run(trl, {inp: np.zeros((2*3*4*123)), pat: np.zeros((2,3,4,1))}).shape
    (2, 3, 4, 123)
    >>> sess.run(trl, {inp: np.zeros((2*3*4*123)), pat: np.zeros((2,3,4,456))}).shape
    (2, 3, 4, 123)
    >>> sess.run(trl, {inp: np.zeros((2*3*4*1)), pat: np.zeros((2,3,4,456))}).shape
    (2, 3, 4, 1)
 
    >>> trl = tf_reshape_like_notlast(inp, pat, squeeze_last=True)
    >>> sess.run(trl, {inp: np.zeros((2*3*4*1)), pat: np.zeros((2,3,4,456))}).shape
    (2, 3, 4)
    '''

    if squeeze_last:
        return tf.squeeze(tf.reshape(tensor, tf_shape_notlast(pattern, append_neg1=True)), [-1], name=name)
    else:
        return tf.reshape(tensor, tf_shape_notlast(pattern, append_neg1=True), name=name)


########################
# Random helpers
########################

def tf_batch_multinomial_with_temperature(logits, temperature=1.0, name=None):
    '''Like tf.multinomial but supports Tensors with or without batch
    dimensions (rank 3+ in addition to rank 2) and sampling based on
    temperature. Temperature can be provided as a Python expression,
    in which case it is evaluated at graph construction time, or a
    Tensor, in which case it is evaluated at run time.

    Hard coded to num_samples = 1!
    
    args:
        logits -- an ND array of shape SHAPE, where N >= 2. Samples are performed across the last dimension.
        Temperature -- usually 0 <= temperature <= 1. (Higher is allowed as well, though not usually used.)
          temperature 0: argmax
          temperature 1: samples from softmax of the logits
          temperature inf: uniform samples
    returns:
        Selected samples from distribution with shape SHAPE[:-1]


    >>> inp = tf.placeholder('float32')
    >>> samples = batch_multinomial_with_temperature(inp, 0)
    >>> sess.run(samples, {inp: np.eye(3)})
    array([0, 1, 2])
    >>> sess.run(samples, {inp: np.eye(3)})
    array([0, 1, 2])

    >>> samples = batch_multinomial_with_temperature(inp, 10.0)
    >>> sess.run(samples, {inp: np.eye(3)})
    array([0, 1, 2])
    >>> sess.run(samples, {inp: np.eye(3)})
    array([2, 2, 0])

    >>> temp = tf.placeholder('float32')
    >>> samples = batch_multinomial_with_temperature(inp, temp)
    >>> sess.run(samples, {inp: np.eye(3), temp:0})
    array([0, 1, 2])
    >>> sess.run(samples, {inp: np.eye(3), temp:0})
    array([0, 1, 2])

    >>> sess.run(samples, {inp: np.eye(3), temp:10})
    array([2, 2, 1])
    >>> sess.run(samples, {inp: np.eye(3), temp:10})
    array([2, 0, 1])
    
    >>> sess.run(samples, {inp: np.zeros((2,3,4,5)), temp:10}).shape
    (2, 3, 4)
    '''

    if not isinstance(temperature, tf.Tensor):
        # zero-temp bool can be evaluated now
        if temperature == 0:
            ret = tf.argmax(logits, tf.rank(logits)-1, name=name)
        else:
            samples = tf.multinomial(tf_flatten_notlast(logits / temperature), num_samples=1)
            ret = tf_reshape_like_notlast(samples, logits, squeeze_last=True, name=name)
    else:
        # zero-temp bool must be deferred to run time
        ret = tf.cond(
            tf.equal(temperature, 0),
            lambda: tf.argmax(logits, tf.rank(logits)-1),
            lambda: tf_reshape_like_notlast(tf.multinomial(tf_flatten_notlast(logits / temperature), num_samples=1), logits, squeeze_last=True),
            name=name
        )
    return ret

def tf_set_intersection(set_a, set_b):
    '''Now with 100% less sparse tensors!
    Like tf.contrib.metrics.set_intersection.
    '''
    assert len(set_a.get_shape()) == len(set_b.get_shape())
    rank_one = len(set_a.get_shape()) == 1
    if rank_one:
        set_a = tf.expand_dims(set_a, 0)
        set_b = tf.expand_dims(set_b, 0)
    intersection = tf.sparse_tensor_to_dense(tf.contrib.metrics.set_intersection(set_a, set_b))
    if rank_one:
        intersection = tf.squeeze(intersection, [0])
    return intersection

def tf_to_bool(x, name=None):
    return tf.cast(x, tf.bool, name=name)

def tf_masked_reduce_mean(tensor, mask, default_val=np.nan, name='masked_mean'):
    '''Computes mean over masked elements of tensor. If all mask elements are False, returns default_val'''

    mask_float = tf.to_float(mask, name='mask_float')
    sum_mask = tf.reduce_sum(mask_float)
    sum_mask_is_zero = tf.equal(sum_mask, 1230.0, name='sum_mask_is_zero')      # HACK
    masked_mean = tf.cond(sum_mask_is_zero,
                          lambda: tf.constant(default_val),
                          lambda: tf.reduce_mean(tf.boolean_mask(tensor, mask)),
                          name=name)
    return masked_mean

########################
# Logging helpers
########################

def get_ptt_names(name):
    '''Gets param/train/test summary names. Just converts, e.g.,
    foo       ->       foo,       foo__train,       foo__test
    scope/foo -> scope/foo, scope/foo__train, scope/foo__test
    '''
    splits = tuple(name.rsplit('/',1))
    if len(splits) == 1:
        return '%s' % name, '%s__train' % name, '%s__test' % name
    else:
        # len 2
        return '%s/%s' % splits, '%s/%s__train' % splits, '%s/%s__test' % splits
    

def normalize_name(name):
    '''Returns a normalized form of name, replacing : with _'''
    return name.replace(':', '_')


def hist_summaries(*args, **kwargs):
    '''Add tf.histogram_summary for each variable in variables'''
    for var in args:
        hist_summary(var, **kwargs)

def hist_summaries_param(*args, **kwargs):
    kwargs['param'] = True
    hist_summaries(*args, **kwargs)

def hist_summaries_traintest(*args, **kwargs):
    kwargs['traintest'] = True
    hist_summaries(*args, **kwargs)

def hist_summaries_train(*args, **kwargs):
    kwargs['train'] = True
    hist_summaries(*args, **kwargs)

def hist_summaries_test(*args, **kwargs):
    kwargs['test'] = True
    hist_summaries(*args, **kwargs)


def hist_summary(var, name=None, traintest=False, param=False, train=False, test=False, orig_collection='orig_histogram'):
    assert sum([int(v) for v in (traintest, param, train, test)]) == 1, 'exactly one of {traintest,train,test,param} should be true'
    if name is None:
        name = var.name
    param_name,train_name,test_name = get_ptt_names(name)
    if traintest:
        train = True
        test = True
    if param:
        tf.summary.histogram(normalize_name(param_name), var, collections=['param_collection', orig_collection])
        #print 'Adding summary.histogram for %s in collections %s, %s' % (var, 'param_collection', orig_collection)
    if train:
        tf.summary.histogram(normalize_name(train_name), var, collections=['train_collection', orig_collection])
        #print 'Adding summary.histogram for %s in collections %s, %s' % (var, 'train_collection', orig_collection)
    if test:
        tf.summary.histogram(normalize_name(test_name), var, collections=['test_collection', orig_collection])
        #print 'Adding summary.histograms for %s in collections %s, %s' % (var, 'test_collection', orig_collection)
        
def scalar_summaries(*args, **kwargs):
    '''Add tf.summary.scalar for each variable in variables'''
    for var in args:
        scalar_summary(var, **kwargs)

def scalar_summaries_param(*args, **kwargs):
    kwargs['param'] = True
    scalar_summaries(*args, **kwargs)

def scalar_summaries_traintest(*args, **kwargs):
    kwargs['traintest'] = True
    scalar_summaries(*args, **kwargs)

def scalar_summary(var, name=None, traintest=False, param=False, also_hist=False, orig_collection='orig_scalar'):
    '''Add tf.summary.scalar for each variable in variables'''
    assert traintest or param, 'one should be true'
    if name is None:
        name = var.name
    param_name, train_name, test_name = get_ptt_names(name)
    if param:
        tf.summary.scalar(normalize_name(param_name), var, collections=['param_collection', orig_collection])
        #print 'Adding summary.scalar for %s in collections %s, %s' % (var, 'param_collection', orig_collection)
    if traintest:
        tf.summary.scalar(normalize_name(train_name), var, collections=['train_collection', orig_collection])
        #print 'Adding summary.scalar for %s in collections %s, %s' % (var, 'train_collection', orig_collection)
        tf.summary.scalar(normalize_name(test_name), var, collections=['test_collection', orig_collection])
        #print 'Adding summary.scalar for %s in collections %s, %s' % (var, 'test_collection', orig_collection)

    if also_hist:
        # HACK: also add hist summary for scalars to show them also on the
        # Histogram pane. Need to provide a unique name so the histogram
        # summary doesn't conflict with the scalar summary
        hist_summary(var, name=normalize_name(name + '_(scalar)'), traintest=traintest, param=param, orig_collection=orig_collection)


def log_scalars(writer, iters, scalars_dict, prefix=None):
    '''Manually log scalar values. Use like this:

    log_scalars(writer, iters, {'test_loss': mean_test_loss,
                                'test_loss_spring': mean_test_loss_spring,
                                'test_loss_cross_ent': mean_test_loss_cross_ent,
                                'test_accuracy': mean_test_acc})
    '''

    if not prefix:
        prefix = ''

    if len(prefix) > 0 and not prefix.endswith('/'):
        prefix = prefix + '/'
        
    for key, val in scalars_dict.items():
        if hasattr(val, 'dtype'):
            val = np.asscalar(val)   # Convert, e.g., numpy.float32 -> float
        writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='%s%s' % (prefix, key), simple_value=val)]).SerializeToString(), iters)


def image_summaries(*args, **kwargs):
    '''Add tf.image_summary for each variable in variables'''
    for var in args:
        image_summary(var, **kwargs)

def image_summaries_param(*args, **kwargs):
    kwargs['param'] = True
    image_summaries(*args, **kwargs)

def image_summaries_traintest(*args, **kwargs):
    kwargs['traintest'] = True
    image_summaries(*args, **kwargs)

def image_summaries_train(*args, **kwargs):
    kwargs['train'] = True
    image_summaries(*args, **kwargs)

def image_summaries_test(*args, **kwargs):
    kwargs['test'] = True
    image_summaries(*args, **kwargs)


def image_summary(var, name=None, traintest=False, param=False, train=False, test=False, orig_collection='orig_image'):
    assert sum([int(v) for v in (traintest, param, train, test)]) == 1, 'exactly one of {traintest,train,test,param} should be true'
    if name is None:
        name = var.name
    param_name,train_name,test_name = get_ptt_names(name)
    if traintest:
        train = True
        test = True
    if param:
        tf.summary.image(normalize_name(param_name), var, collections=['param_collection', orig_collection])
        #print 'Adding image_summary for %s in collections %s, %s' % (var, 'param_collection', orig_collection)
    if train:
        tf.summary.image(normalize_name(train_name), var, collections=['train_collection', orig_collection])
        #print 'Adding image_summary for %s in collections %s, %s' % (var, 'train_collection', orig_collection)
    if test:
        tf.summary.image(normalize_name(test_name), var, collections=['test_collection', orig_collection])
        #print 'Adding image_summary for %s in collections %s, %s' % (var, 'test_collection', orig_collection)
        

def add_grads_and_vars_hist_summaries(grads_and_vars):
    '''Adds param summary of var and hist summaries of grad values for
    the given list of (grad,var) tuples. Usually these tuples will
    come from the optimizer, e.g. via:

    grads_and_vars = opt_rest.compute_gradients(model.v.loss, model.trainable_weights())
    '''

    for grad, var in grads_and_vars:
        if grad is None:
            continue
        grad_name = '%s__grad' % var.name
        hist_summaries_train(grad, name=grad_name)
        hist_summaries_param(var)


def add_grad_summaries(grads_and_vars, add_summaries_train=True, quiet=False):
    # Add summary nodes for grad values and prints a summary as well
    if not quiet:
        print('\nGrads:')
        if len(grads_and_vars) == 0:
            print('  <None>')
    for grad, var in grads_and_vars:
        if grad is None:
            continue   # skip grads that are None (corner case: not computed because model.loss has no dependence?)
        grad_name = '%s/%s__grad' % tuple(var.name.rsplit('/', 1))
        if not quiet:
            print('  ', grad_name, grad)
        if add_summaries_train:
            hist_summaries_train(grad, name=grad_name)
    if not quiet:
        print()


########################
# TF operation helpers
########################

def hacked_tf_one_hot(indices, depth, on_value, off_value, name=None):
    '''Emulates new tf.one_hot in master.
    # Real signature:    tf.one_hot(indices, depth, on_value, off_value, axis=None, name=None)
    # Assumed signature: tf.one_hot(indices, depth, on_value, off_value, axis=-1,   name=None)

    Not needed if using newer versions of TensorFlow.
    '''
    
    N = tf.shape(indices)[0]
    range_Nx1 = tf.expand_dims(tf.to_int64(tf.range(N)), 1)
    indices_Nx1 = tf.expand_dims(indices, 1)
    concat = tf.concat(1, [range_Nx1, indices_Nx1])
    as_dense = tf.sparse_to_dense(concat,
                                  tf.to_int64(tf.pack([N, depth])), # Assumption: axis=-1
                                  on_value, off_value)
    one_hot = tf.reshape(as_dense, (-1, depth), name=name)
    
    return one_hot


def hacked_tf_nn_softmax(logits, name=None):
    '''Like tf.nn.softmax but casts to float64 first as a workaround for this bug:
    https://github.com/tensorflow/tensorflow/issues/4425
    '''

    logits_64 = tf.cast(logits, tf.float64)
    out_64 = tf.nn.softmax(logits_64)
    out_32 = tf.cast(out_64, tf.float32, name=name)
    return out_32


def smooth_l1(x, name=None):
    '''Pointwise smooth abs function'''
    absx = tf.abs(x)
    big = tf.cast(tf.greater(absx, tf.ones_like(absx)), tf.float32)
    activation = tf.add(tf.mul(big, absx-.5), tf.mul((1-big), .5*tf.square(x)), name=name)
    return activation
    

########################
# Misc helpers
########################
        
def sess_run_dict(sess, fetch_names, fetch_vars=None, feed_dict=None, options=None, run_metadata=None, **other_kwargs):
    '''
    Like sess.run but returns a dictionary of results
    Usage:
    sess_run_dict(sess, fetch_names, fetch_vars, ...)
    sess_run_dict(sess, fetch_dict, ...)
    '''
    
    dict_mode = isinstance(fetch_names, dict)

    if dict_mode:
        assert fetch_vars is None, 'provide either dict or list of names and vars, not both'
        fetch_dict = fetch_names
        fetch_names = list(fetch_dict.keys())
        fetch_vars = list(fetch_dict.values())
    
    assert len(fetch_names) == len(fetch_vars), 'length of fetch_names must match length of fetch_vars'
    assert isinstance(fetch_vars, list) or isinstance(fetch_vars, tuple), 'fetch_vars should be list or tuple'
    result = sess.run(fetch_vars, feed_dict=feed_dict, options=options, run_metadata=run_metadata, **other_kwargs)
    ret = {k:v for k,v in zip(fetch_names, result)}
    return ret


def get_collection_intersection(*args):
    ret = []
    for ii,arg in enumerate(args):
        if ii == 0:
            ret = OrderedSet(tf.get_collection(arg))
        else:
            ret = ret.intersection(OrderedSet(tf.get_collection(arg)))
    return list(ret)


def get_collection_intersection_summary(*args):
    '''Returns a tf.merge_summary of the given collection intersection, or None if the intersection is empty.'''
    col_int = get_collection_intersection(*args)
    if col_int:
        return tf.summary.merge(col_int)


def summarize_weights(weights, sess=None):
    '''Print summary of each weight tensor in a list of weight tensors.
    Example usage:
    summarize_weights(model.trainable_weights)

    if sess is provided, also print weight min, max, and RMS
    '''

    if sess:
        vals = sess.run(weights)
    total_params = 0
    titl = '  %50s: %10s %-20s' % ('NAME', 'SIZE', 'SHAPE')
    if sess:
        titl += ' %10s, %10s, %10s' % ('MIN', 'MAX', 'RMS')
    print(titl)
    for ii,var in enumerate(weights):
        st = '  %50s: %10d %-20s' % (var.name, np.prod(var.get_shape().as_list()), var.get_shape().as_list())
        if sess:
            val = vals[ii]
            st += ' %10s, %10s, %10s' % ('%.3g' % val.min(), '%.3g' % val.max(), '%.3g' % np.sqrt((val**2).mean()))
        print(st)
        total_params += np.prod(var.get_shape().as_list())
    print('  %50s: %10d' % ('Total', total_params))
    return total_params


def val_or_dynamic(vord):
    return '<dynamic>' if isinstance(vord, tf.Tensor) else repr(vord)


def summarize_opt(opt):
    print('Optimizer:')
    print('   ', opt)
    if isinstance(opt, tf.train.MomentumOptimizer):
        print('   LR: %s, momentum: %g, use_nesterov: %s' % (val_or_dynamic(opt._learning_rate), opt._momentum, opt._use_nesterov))
    elif isinstance(opt, tf.train.RMSPropOptimizer):
        print('   LR: %s, momentum: %g, decay: %g, epsilon: %g' % (val_or_dynamic(opt._learning_rate), opt._momentum, opt._decay, opt._epsilon))
    elif isinstance(opt, tf.train.AdamOptimizer):
        print('   LR: %s, beta1: %g, beta2: %g, epsilon: %g' % (val_or_dynamic(opt._lr), opt._beta1, opt._beta2, opt._epsilon))
    else:
        print('   (cannot summarize unknown type of optimizer)')


def tf_assert_gpu(sess):
    with tf.device('/gpu:0'):
        foo = tf.placeholder(tf.float32, name='assert_gpu')
        bar = tf.add(foo, 1, name='assert_gpu')
    try:
        sess.run(bar, {foo: 1})
    except:
        print('\n\n\ntf_assert_gpu: no GPU is present! In case it helps, CUDA_VISIBLE_DEVICES is %s' % repr(os.environ.get('CUDA_VISIBLE_DEVICES', None)))
        print('See error below:\n\n\n')
        raise


def tf_assert_all_init(sess):
    uninit_vars = sess.run(tf.report_uninitialized_variables())
    assert len(uninit_vars) == 0, 'Expected all variables to have been initialized, but these have not been: %s' % uninit_vars


def tf_get_uninitialized_variables(sess):
    '''A bit of a hack from
    https://stackoverflow.com/questions/35164529/in-tensorflow-is-there-any-way-to-just-initialize-uninitialised-variables
    to get a list of all uninitialized Variable objects from the
    graph
    '''

    uninitialized_vars = []
    for var in tf.global_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)
    return uninitialized_vars
