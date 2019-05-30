# Util functions for tf_plus

import os
import tensorflow as tf
import numpy as np
from collections import OrderedDict

from general.tfutil import tf_assert_gpu


def deduplist(lst):
    # From https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-in-whilst-preserving-order
    seen = set()
    seen_add = seen.add
    ret = [x for x in lst if not (x in seen or seen_add(x))]
    return ret


def uses_learning_phase(layer):
    '''A Layer uses the learning_phase if it has the _uses_learning_phase attribute and it is set to True.'''
    if hasattr(layer, 'uses_learning_phase'):
        return layer.uses_learning_phase
    elif hasattr(layer, '_uses_learning_phase'):
        return layer._uses_learning_phase
    else:
        return False


class ObjectWithGroups(object):
    def __init__(self, obj, groups=None):
        self.obj = obj
        if groups is None:
            groups = set()
        assert isinstance(groups, set)
        self.groups = groups


class NamedObjectStore(object):
    '''Stores named objects'''

    def __init__(self):
        '''Storage format:
        dict[name] = list of ObjectWithGroups
        List: one per time objects with this name have been added.

        Name is always stored in the singular form (without an extra "s" suffix).
        '''
        self._objects = OrderedDict()

    def add(self, name, obj, groups, allow_multiple=False):
        assert len(name) > 0, 'Name cannot be blank'
        plural_name = name + 's'
        assert plural_name not in self._objects, 'Cannot add singular name "%s" because plural "%ss" already exists' % (name, name)
        if name[-1] == 's':
            assert not name[:-1] in self._objects, 'Cannot add name "%s" because singular form "%s" already exists' % (name, name[:-1])
        if not allow_multiple and name in self._objects:
            raise Exception('Cannot add another object with name "%s" to NamedObjectStore because it already contains one and allow_multiple is False')

        assert isinstance(groups, set), 'Argument groups should be a set'

        if name not in self._objects:
            self._objects[name] = []

        self._objects[name].append(ObjectWithGroups(obj, set(groups)))

        plural_attr = [x.obj for x in self._objects[name]]
        if len(self._objects[name]) > 1:
            # Hack to disable single_attr access with a helpful error
            singular_attr = 'There are multiple objects defined using name "%s", so access using %s[index] instead' % (name, plural_name)
        else:
            singular_attr = plural_attr[0]
        return singular_attr, plural_attr

    def names(self):
        return list(self._objects.keys())
        
    def items(self):
        return list(self._objects.items())
        
    def name_exists(self, name):
        return name in self._objects
        
    def get_object_with_groups(self, name):
        assert len(name) > 0, 'Name cannot be blank'
        if name in self._objects:
            # singular form requested, so return a single one
            assert len(self._objects[name]) == 1, 'Cannot request object "%s" because there are multiple objects with that name. Specify which one instead by requesting "%ss" and slicing.' % (name, name)
            return self._objects[name][0]
        elif name[-1] == 's' and name[:-1] in self._objects:
            # plural form requested, so return all
            return self._objects[name[:-1]]
        else:
            raise KeyError('Object(s) with name %s not found in store.' % name)

    def get(self, name):
        object_with_groups = self.get_object_with_groups(name)
        if isinstance(object_with_groups, list):
            return [x.obj for x in object_with_groups]
        else:
            return object_with_groups.obj

def setup_session_and_seeds(seed, assert_gpu=True, mem_fraction=None):
    '''Start TF session'''

    # Use InteractiveSession instead of Session so the default session will be set
    if mem_fraction is not None:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_fraction)
        sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    else:
        sess = tf.InteractiveSession()
    np.random.seed(seed)
    tf.set_random_seed(seed)
    print('Set numpy and tensorflow random seeds to: %s' % repr(seed))
    print('My PID is %d' % os.getpid())
    if assert_gpu:
        tf_assert_gpu(sess)
    return sess

def print_trainable_warnings(model, graph=None):
    '''Print warnings for any vars marked as trainable in the
    model but not graph, and vice versa. A common case where this
    occurs is in BatchNormalization layers, where internal
    variables are updated but not marked as trainable.
    '''

    if graph is None:
        try:
            graph = tf.python.get_default_graph()
        except AttributeError:
            graph = tf.get_default_graph()

    def tag(name):
        if 'batchnormalization' in name and 'running' in name:
            # Keras 1.2.2
            return ' . '
        elif 'batch_normalization' in name and 'moving' in name:
            # Keras 2+
            return ' . '
        else:
            return '***'

    # Check which vars are trainable
    trainable_vars_from_graph = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    trainable_vars_from_model = model.trainable_weights

    in_graph_not_model = set(trainable_vars_from_graph).difference(set(trainable_vars_from_model))
    if in_graph_not_model:
        print('Warning: the following vars are marked as trainable in the graph but not in model.trainable_weights (typical for BatchNormalization layers. "." if expected, "***" if not):')
        print('\n'.join(['   %4s %s: %s' % (tag(vv.name), vv.name, vv) for vv in in_graph_not_model]))
    in_model_not_graph = set(trainable_vars_from_model).difference(set(trainable_vars_from_graph))
    if in_model_not_graph:
        print('Warning: the following vars are in model.trainable_weights but not marked as trainable in the graph:')
        print('\n'.join(['   %4s %s: %s' % (tag(vv.name), vv.name, vv) for vv in in_model_not_graph]))
