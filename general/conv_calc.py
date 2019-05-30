#! /usr/bin/env python

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


# Functions for calculating convnet receptive field sizes
# Can use, e.g., like this:
# rso = [(1,1,0)]                  Input layer receptive field size (1 pixel), stride (1 pixel), and offset (0 pixels)
# rso.append(receptive_calc_conv(5, 1)(rso[-1]))
# rso.append(receptive_calc_pool(2, 2)(rso[-1]))
# rso.append(receptive_calc_conv_same(5, 1)(rso[-1]))
# rso.append(receptive_calc_conv_same(5, 1)(rso[-1]))
# print rso
# [(1, 1, 0), (5, 1, 0), (6, 2, 0), (14, 2, -4), (22, 2, -8)]
# print rso[-1]
# (22, 2, -8)                      Output layer receptive field size (22 pixels), stride (2 pixels), and offset (-8 pixels)

import numpy as np


def receptive_calc_conv(filt_hw, filt_stride):
    '''Return a function from rso -> rso. rso is a tuple of (receptive field hw, stride, and offset). Assumes 'valid' convolution.'''
    def fn(rso):
        rec_hw, rec_stride, rec_offset = rso
        return (
            rec_hw + rec_stride*(filt_hw-1),
            rec_stride * filt_stride,
            rec_offset
        )
    return fn

# Same function works for pool
receptive_calc_pool = receptive_calc_conv

def receptive_calc_conv_same(filt_hw, filt_stride):
    '''Return a function from rso -> rso. rso is a tuple of (receptive field hw, stride, and offset). Assumes 'same' convolution.'''
    assert np.mod(filt_hw, 2) == 1, 'receptive_calc_conv_same only works for convolution with odd-sized filters'
    def fn(rso):
        rec_hw, rec_stride, rec_offset = rso
        return (
            rec_hw + rec_stride*(filt_hw-1),
            rec_stride * filt_stride,
            rec_offset - (filt_hw-1)/2 * rec_stride
        )
    return fn

def receptive_calc_pad(pad_hw):
    def fn(rso):
        rec_hw, rec_stride, rec_offset = rso
        return (
            rec_hw,
            rec_stride,
            rec_offset - pad_hw*rec_stride,
        )
    return fn
