# Copyright (c) 2019 Uber Technologies, Inc.
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

import argparse
from general.util import string_or_gitresman_or_none

DEFAULT_ARCH_CHOICES = ['mnist']


def make_standard_parser(description='No decription provided', arch_choices=DEFAULT_ARCH_CHOICES,
                         skip_train=False, skip_val=False):
    '''Make a standard parser, probably good for many experiments.

    Arguments:

      description: just used for help

      arch_choices: list of strings that may be specified when
         selecting architecture type. For example, ('mnist', 'cifar')
         would allow selection of different networks for each
         dataset. architecture may also be toggled via the --conv and
         --xprop switches. Default architecture is the first in the
         list.

      skip_train: if True, skip adding a train_h5 arg

      skip_val: if True, skip adding a val_h5 arg
    '''
    
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog)
    )

    # Optimization
    parser.add_argument('--opt', type=str, default='sgd', choices=('sgd', 'rmsprop', 'adam'), help='Which optimizer to use')
    parser.add_argument('--lr', '-L', type=float, default=.001, help='learning rate')
    parser.add_argument('--mom', '-M', type=float, default=.9, help='momentum (only has effect for sgd/rmsprop)')
    parser.add_argument('--beta1', type=float, default=.9, help='beta1 for adam opt')
    parser.add_argument('--beta2', type=float, default=.99, help='beta2 for adam opt')
    parser.add_argument('--adameps', type=float, default=1e-8, help='epsilon for adam opt')
    parser.add_argument('--epochs', '-E',type=int, default=5, help='number of epochs.')

    # Model
    parser.add_argument('--arch', type=str, default=arch_choices[0],
                        choices=arch_choices, help='Which architecture to use (choices: %s).' % arch_choices)
    parser.add_argument('--conv', '-C', action='store_true', help='Use a conv model.')
    parser.add_argument('--xprop', '-X', action='store_true', help='Use an xprop model')
    parser.add_argument('--springprop', '-S', action='store_true', help='Use an springprop model')
    parser.add_argument('--springt', '-t', type=float, default=0.5, help='T value to use for springs')
    parser.add_argument('--learncoords', '--lc', action='store_true', help='Learn coordinates (update them during training) instead of keeping them fixed.')
    parser.add_argument('--l2', type=float, default=0.0, help='L2 regularization to apply to direct parameters.')
    parser.add_argument('--l2i', type=float, default=0.0, help='L2 regularization to apply to indirect parameters.')

    # Experimental setup
    parser.add_argument('--seed', type=int, default=0, help='random number seed for intial params and tf graph')
    parser.add_argument('--minibatch', '-mb', type=int, default=256, help='minibatch size')
    parser.add_argument('--test', action='store_true', help='Use test data instead of validation data (for final run).')
    parser.add_argument('--shuffletrain', '--st', dest='shuffletrain', action='store_true', help='Shuffle training set each epoch.')
    parser.add_argument('--noshuffletrain', '--nst', dest='shuffletrain', action='store_false', help='Do not shuffle training set each epoch. Ignore the following "default" value:')
    parser.set_defaults(shuffletrain=True)
    
    # Misc
    parser.add_argument('--ipy', '-I', action='store_true', help='drop into embedded iPython for debugging.')
    parser.add_argument('--nocolor', '--nc', action='store_true', help='Do not use color output (for scripts).')
    parser.add_argument('--skipval', action='store_true', help='Skip validation set entirely.')
    parser.add_argument('--verbose', '-V', action='store_true', help='Verbose mode (print some extra stuff)')
    parser.add_argument('--cpu', action='store_true', help='Skip GPU assert (allows use of CPU but still uses GPU if possible)')

    # Saving a loading
    parser.add_argument('--snapshot-to', type=str, default='net', help='Where to snapshot to. --snapshot-to NAME produces NAME_iter.h5 and NAME.json')
    parser.add_argument('--snapshot-every', type=int, default=-1, help='Snapshot every N minibatches. 0 to disable snapshots, -1 to snapshot only on last iteration.')
    parser.add_argument('--load', type=str, default=None, help='Snapshot to load from: specify as H5_FILE:MISC_FILE.')
    parser.add_argument('--output', '-O', type=string_or_gitresman_or_none, default='', help='directory output TF results to. If None, checks for GIT_RESULTS_MANAGER_DIR environment variable and uses that directory, if defined, unless output is set to "skip", in which case no output is written even if GIT_RESULTS_MANAGER_DIR is defined. If nothing else: skips output.')

    # Dataset
    if not skip_train:
        parser.add_argument('train_h5', type=str, help='Training set hdf5 file.')
    if not skip_val:
        parser.add_argument('val_h5', type=str, help='Validation set hdf5 file.')

    return parser
import numpy as np

def merge_dict_append(d1,d2):
    assert d1.keys() == d2.keys(), 'Two dictionaries to merge must have the same set of keys'

    merged = {}
    for kk in d1.keys():
        v1 = d1[kk] if isinstance(d1[kk], list) else [d1[kk]]
        v2 = d2[kk] if isinstance(d2[kk], list) else [d2[kk]]
        merged[kk] = v1 + v2 
    return merged

def average_dict_values(dd):
    averaged = {}
    for kk in dd.keys():
        assert len(dd[kk]) == len(dd['weights']), "lengths must match"
        x1 = np.array(dd[kk])
        x2 = np.array(dd['weights'])
        averaged[kk] = np.sum(np.multiply(x1,x2)) / np.sum(x2)
    return averaged


def interpolate(z0, z1, t):
    # z0: noise vector 0
    # z1: noise vector 1
    # t:  scale between [0,1]
    r0 = np.linalg.norm(z0)
    r1 = np.linalg.norm(z1)

    rscale = r0 * t + r1 * (1-t)
    zt = z0 * t + z1 * (1-t)
    rt = np.linalg.norm(zt)

    zt /= (rt/rscale)

    return zt

def image_separator(consolidated_images, nh=10, nw=10):
    # inverse function of exp.cgan.utils.merge  
    h = (consolidated_images.shape[0]-nh+1) / nh
    w = (consolidated_images.shape[1]-nw+1) / nw
    gray = True if len(consolidated_images.shape) == 2 else False
    images = []
    for ii in range(nh):
        for jj in range(nw):
            start_w = ii*w if ii == 0 else ii*(w+1)
            start_h = jj*h if jj == 0 else jj*(h+1)
            if gray:
                _image = consolidated_images[start_h:start_h + h, start_w:start_w + w]
            else:
                _image = consolidated_images[start_h:start_h + h, start_w:start_w + w, :]
            images.append(_image)
    return images
