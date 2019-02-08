from collections import OrderedDict

import tensorflow as tf
from tensorflow.python.layers import base
from tensorflow.python.layers.normalization import BatchNormalization as base_BatchNormalization

from general.util import DotDict
from .util import deduplist, uses_learning_phase, NamedObjectStore
#from .backend import learning_phase


class BaseLayer(base.Layer):
    '''Minimal extensions to tf.layers.Layer necessary to track activations and other tensors. Abstract base class.'''

    def __init__(self, *args, **kwargs):
        super(BaseLayer, self).__init__(*args, **kwargs)
        self._named_objects = NamedObjectStore()

    def _add_named_object(self, name, object_or_list, groups=None, allow_multiple=True):
        '''Allow addition of named objects or lists of objects. Typical named objects will include
          - Named activations. "input" and "output" are always added, and additional activations can be added per layer, usually in the call function.
          - Named trainable or non-trainable weights, usually in the build function.
          - Named sublayers (applicable to the Layers class and derived subclasses).
          - Etc?

        Objects may be added to one or more groups addressed by arbitrary strings.

        <multiple objects added...>
        '''

        group_set = groups or set()
        if not isinstance(group_set, set):
            group_set = set(group_set)

        #assert not allow_multiple, 'do later'

        if hasattr(self, name) and not self._named_objects.name_exists(name):
            # Attr already defined but not via _add_named_object.
            raise Exception('Cannot add a named object with name "%s" because this attribute is already defined' % name)
        singular_attr, plural_attr = self._named_objects.add(name, object_or_list, group_set, allow_multiple=allow_multiple)
        setattr(self, name, singular_attr)
        setattr(self, name + 's', plural_attr)
        return object_or_list

    def add_named_act(self, name, object_or_list, groups=None, trackable=False):
        groups = groups or []
        if trackable and 'trackable' not in groups:
            groups.append('trackable')
        return self._add_named_object(name, object_or_list, groups=groups + ['act'])

    def add_named_weight(self, name, object_or_list, groups=None):
        groups = groups or []
        return self._add_named_object(name, object_or_list, groups=groups + ['weight'], allow_multiple=False)

    # Short aliases for add_named_{act,weight}
    a = add_named_act
    w = add_named_weight

    def get(self, name):
        '''Fetches the given named object. Singular (NAME) or plural form
        (NAMEs) may be requested, but requesting the singular NAME is
        an error if multiple objects with that name exist.
        '''
        return self._named_objects.get(name)

    def named_keys(self):
        '''Fetches the given named object. Singular (NAME) or plural form
        (NAMEs) may be requested, but requesting the singular NAME is
        an error if multiple objects with that name exist.
        '''
        return self._named_objects.names()

    def summarize_named(self, include_groups=None, prefix='', final_prefix=''):
        '''Print a summary of named objects, optionally filtered to only include the given group(s)'''
        if include_groups is not None and not (isinstance(include_groups, list) or isinstance(include_groups, tuple)):
            include_groups = [include_groups]
        for name in self._named_objects.names():
            plural_name = name + 's'
            owgs = self._named_objects.get_object_with_groups(plural_name)
            for ii, owg in enumerate(owgs):
                if len(owgs) > 1:
                    usename = '%s[%d]' % (plural_name, ii)
                else:
                    usename = name
                include = True
                if include_groups is not None:
                    for group in include_groups:
                        if group not in owg.groups:
                            include = False
                if include or hasattr(owg.obj, 'summarize_named'):
                    objstr = str(owg.obj)
                    objstr = objstr if len(objstr) < 70 else '%s...' % objstr[:67]
                    objstr = objstr.replace('\n', '')
                    st = '%s%s%s: %s' % (prefix, final_prefix, usename, objstr)
                    if len(owg.groups) > 0:
                        st += ' (%s)' % (', '.join([str(gg) for gg in owg.groups]))
                    print st
                if hasattr(owg.obj, 'summarize_named'):
                    owg.obj.summarize_named(include_groups=include_groups, prefix=prefix + '  ', final_prefix='.')

    def named_dict(self, include_groups=None, name_prefix='', recursive=True):
        '''Return an OrderedDict of {name: object}, optionally filtered to include only the given group(s).'''
        if include_groups is not None and not (isinstance(include_groups, list) or isinstance(include_groups, tuple)):
            include_groups = [include_groups]
        ret = OrderedDict()
        for name, owgs in self._named_objects.items():
            #print 'TODO NEXT: fix the problem where, say, trackables under unnamed layers are not returned'
            plural_name = name + 's'
            for ii, owg in enumerate(owgs):
                if len(owgs) > 1:
                    usename = '%s:%d' % (plural_name, ii)
                else:
                    usename = name
                include = True
                if include_groups is not None:
                    for group in include_groups:
                        if group not in owg.groups:
                            include = False
                if include:
                    ret[usename] = owg.obj
                    if isinstance(owg.obj, list):
                        raise Exception('TODO: deal with lists of tensors.')
                if recursive and hasattr(owg.obj, 'named_dict'):
                    subret = owg.obj.named_dict(include_groups=include_groups, name_prefix='%s/' % usename)
                    ret.update(subret)
        return ret

    def obj_list(self, include_groups=None, recursive=True):
        return self.named_dict(include_groups=include_groups, recursive=recursive).values()

    def own_obj_list(self, include_groups=None):
        return self.named_dict(include_groups=include_groups, recursive=False).values()

    def trackable_dict(self):
        return self.named_dict(include_groups='trackable')

    def trackable_names(self):
        return self.named_dict(include_groups='trackable').keys()

    def update_dict(self):
        return OrderedDict(('update__%d' % ii, update) for ii, update in enumerate(self.updates))

    def trackable_and_update_dict(self):
        '''Returns a dict of all trackables and updates. Useful for
        training when you want to fetch all trackables and also ensure
        any updates (e.g. for rolling average BatchNormalization
        layers) are fetched.
        '''

        ret = self.trackable_dict()
        ret.update(self.update_dict())
        return ret

    def __call__(self, input_, *args, **kwargs):
        # Add underscored _input and _output to avoid conflict with Keras layers' input(s) and output(s)
        self.add_named_act('_input', input_)
        output = super(BaseLayer, self).__call__(input_, *args, **kwargs)
        self.add_named_act('_output', output)
        return output


class Layers(BaseLayer):
    '''Minimal extensions to BaseLayer necessary to allow containing sublayers. Abstract base class.

    Inspired by Keras Model and base.Network (but not called on inputs
    and outputs) and tfe.Network (but without some of the complexity and work-in-progress-ness).

    + learning_phase aware
    '''

    def __init__(self, uses_learning_phase=False, *args, **kwargs):
        super(Layers, self).__init__(*args, **kwargs)
        self._uses_learning_phase = uses_learning_phase

    def track_layer(self, name, layer=None, groups=None):
        '''JBY: Inspired from tfe.Network, but with a slightly different
        interface (different arg order, allows tracking with arbitrary
        extra groups.

        Track a Layer in this Layers.

        `Layers` requires that all `Layer`s used in `call()` be tracked so that the
        `Layers` can export a complete list of variables.

        Args:
          name: name so that layer can be accessed with layers.name. If None, an underscore prefixed name will be generated.
          layer: A `tf.layers.Layer` object.

          If only one arg is provided, it is take as the layer with no name.

        Returns:
          The passed in `layer`.

        Raises:
          TypeError: If `layer` is the wrong type.
          ValueError: If a `Layer` with the same name has already been added.
        '''

        if layer is None:
            layer = name
            name = None
        
        if not isinstance(layer, base.Layer):
            raise TypeError(
                'Layers.track_layer() passed type %s, not a tf.layers.Layer' %
                (type(layer),))
        
        if name is None:
            ii = 0
            while name is None and ii < 10000:
                try_name = '_layer_%d' % ii
                if hasattr(self, try_name):
                    ii += 1
                else:
                    name = try_name
            assert name is not None, 'Could not find unique auto-generated name, last tried: %s' % try_name

        return self._add_named_layer(name, layer, groups=groups)
    
    # Short aliases for track_layer
    l = track_layer

    def _add_named_layer(self, name, layer, groups=None):
        '''Note: generally do NOT call this function directly; instead, use track_layer above and provide a name argument.'''
        groups = groups or []
        if 'layer' not in groups:
            groups = groups + ['layer']
        return self._add_named_object(name, layer, groups=groups, allow_multiple=False)

    @property
    def layers(self):
        return self.own_obj_list('layer')

    @property
    def uses_learning_phase(self):
        ret = self._uses_learning_phase
        for layer in self.layers:
            ret |= uses_learning_phase(layer)
        return ret

    @property
    def trainable_weights(self):
        ret = super(Layers, self).trainable_weights
        for layer in self.layers:
            ret.extend(layer.trainable_weights)
        return deduplist(ret)

    @property
    def non_trainable_weights(self):
        ret = super(Layers, self).non_trainable_weights
        for layer in self.layers:
            ret.extend(layer.non_trainable_weights)
        return deduplist(ret)

    @property
    def updates(self):
        ret = super(Layers, self).updates
        for layer in self.layers:
            ret.extend(layer.updates)
        return deduplist(ret)

    @property
    def losses(self):
        ret = super(Layers, self).losses
        for layer in self.layers:
            ret.extend(layer.losses)
        return deduplist(ret)

    def call(self, inputs):
        raise Exception('Implement in derived class')


class SequentialNetwork(Layers):
    '''Sequential version of Layers.'''

    def __init__(self, layers=None, *args, **kwargs):
        super(SequentialNetwork, self).__init__(*args, **kwargs)
        self._called = False
        if layers is not None:
            for layer in layers:
                self.add(layer)

    def add(self, layer_or_tuple, name=None):
        '''layer_or_tuple: may be a layer or a tuple of ('name', layer) to add a named layer.'''
        assert not self._called, 'Cannot add layers after USequentialNetwork has been called on an input'
        if isinstance(layer_or_tuple, tuple) or isinstance(layer_or_tuple, list):
            assert name is None, 'Provide name either as first element of tuple or via name argument, not both'
            name = layer_or_tuple[0]
            layer = layer_or_tuple[1]
        else:
            layer = layer_or_tuple
        self.track_layer(name, layer)

    def addn(self, list_of_layer_or_tuple):
        '''layer_or_tuple: may be a layer or a tuple of ('name', layer) to add a named layer.'''
        for item in list_of_layer_or_tuple:
            self.add(item)

    def build(self, *args, **kwargs):
        super(SequentialNetwork, self).build(*args, **kwargs)

    def call(self, inputs):
        #assert False
        self._called = True
        ret = inputs
        for ii, layer in enumerate(self.layers):
            # Hack / for now: check for BatchNormalization layers
            if issubclass(layer.__class__, base_BatchNormalization):
                assert layer.__class__ != base_BatchNormalization, 'Must use BatchNormalization from tf_plus rather than tf.layers.BatchNormalization'
            ret = layer(ret)
        return ret
