#! /usr/bin/env python

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


import os
import sys
import _pickle as pickle
import gzip
import numpy as np
from PIL import Image
from IPython import embed

from general.vision import ImagePreproc
import tensorflow as tf
mnist = tf.keras.datasets.mnist

def random_position(rng, canvas_size):
    '''Generate a random position that fits in the canvas.'''
    min_hw = 15
    max_hw = 25
    height = rng.uniform(min_hw, max_hw)
    width = rng.uniform(min_hw, max_hw)

    i_center = rng.uniform(height/2, canvas_size[0]-height/2)
    j_center = rng.uniform(width/2,  canvas_size[1]-width/2)

    return i_center, j_center, height, width


def generate_dataset(raw_x, raw_y, n_per, seed, canvas_size = (64,64), center_crop=False, crop_size=(64,64)):
    if len(raw_x.shape) == 4:
        raw_x = raw_x.squeeze()   # Convert (b,1,h,w) or (b,h,w,1) to (b,h,w)
    rng = np.random.RandomState(seed)
    n_ims = raw_x.shape[0]
    perms = [rng.permutation(n_ims) for _ in xrange(n_per)]  # which digits will be included in each image

    # Example shapes for n_ims = 50000, n_per = 5, canvas_size = (64,64)
    # e.g. (50000, 64, 64)           -> reshape at end
    ret_ims = np.zeros((n_ims, canvas_size[0], canvas_size[1]))
    # e.g. (50000, 5, 4)
    ret_pos = np.array([random_position(rng, canvas_size) for _ in xrange(n_ims * n_per)])
    ret_pos = ret_pos.reshape((n_ims, n_per, -1))
    # e.g. (50000, 5)
    ret_class = np.array([[raw_y[perm[ii]] for ii in xrange(len(perm))] for perm in perms]).T
    
    for oo in xrange(n_ims):
        canvas = ret_ims[oo]
        for pp in xrange(n_per):
            # transform and paint
            digit_idx = perms[pp][oo]
            digit_full = raw_x[digit_idx]
            imin = np.where(digit_full.sum(1) > .001)[0][0]
            imax = np.where(digit_full.sum(1) > .001)[0][-1]
            jmin = np.where(digit_full.sum(0) > .001)[0][0]
            jmax = np.where(digit_full.sum(0) > .001)[0][-1]
            digit_crop = digit_full[imin:imax+1, jmin:jmax+1]    # digit less zero borders, shape somethign like (20,18)

            # get target position of digit
            i_center, j_center, height, width = ret_pos[oo][pp]     # target position in canvas (float)
            #min_i_int = int(i_center - height/2)
            #max_i_int = int(i_center + height/2)+1
            #min_j_int = int(j_center - width/2)
            #max_h_int = int(j_center + width/2)+1            
            #assert min_i_int >= 0
            #assert min_j_int >= 0
            #assert max_i_int < canvas.shape[0], 'sample height too large?'
            #assert max_j_int < canvas.shape[1], 'sample width too large?'
            #mini_canv_h = max_i_int-min_i_int+1
            #mini_canv_w = max_j_int-min_j_int+1

            h_scale = height / digit_crop.shape[0]      # < 1.0 to shrink
            w_scale = width / digit_crop.shape[1]
            i_offset = i_center - height/2
            j_offset = j_center - width/2
            #crop_coord_i_min = 
            #min_i_coord =
            #(i_center - height/2 - min_i_int) / (max_i_int - min_i_int)

            imt = Image.fromarray(digit_crop).transform(
                canvas_size,
                Image.AFFINE,
                (1.0/w_scale, 0, -j_offset/w_scale, 0, 1.0/h_scale, -i_offset/h_scale),
                resample=Image.BILINEAR
            )
            canvas += np.array(imt)
            
    ret_ims = np.expand_dims(ret_ims, -1)       # eg (50000, 28, 28) -> (50000, 28, 28, 1)
    # Because previous images were added, now some pixels have higher value than 1.0. Crop them to 1.0
    ret_ims = np.minimum(ret_ims, 1.0)

    if center_crop:
        assert crop_size[0] <= canvas_size[0] and crop_size[1] <= canvas_size[1], "specified crop_size bigger than canvas"
        impreproc = ImagePreproc()
        ret_ims, off_h, off_w = impreproc.center_crops(ret_ims, crop_size, ret_offsets=True)
        # ret_ims_crop = ret_ims[:, off_h:off_h+crop_size[0], off_w:off_w+crop_size[1], :]
        ret_pos[:,:,0] -= off_h
        ret_pos[:,:,1] -= off_w

    return ret_ims, ret_pos, ret_class


def load_tvt_n_per_field(n_per):
    '''
    array ims:
      - with dtype float32
      - values in the range 0 to 1
      - shapes like (50000, 28, 28, 1)
    array pos:
      - with dtype float32
      - with values in the range 0 to 28    (i_center, j_center, h, w)
      - shapes like (50000, 5, 4)           for train and n = 5
    array class:
      - with dtype uint8
      - with values in the range 0 to 1
      - shapes like (50000, 5, )            for train and n = 5
    '''

    raw_train_x, raw_train_y, raw_test_x, raw_test_y = mnist.load_data()
    
    train_ims, train_pos, train_class = generate_dataset(raw_train_x, raw_train_y, n_per, 0)
    test_ims,  test_pos,  test_class  = generate_dataset(raw_test_x,  raw_test_y,  n_per, 1)

    return train_ims, train_pos, train_class, test_ims, test_pos, test_class



def load_tvt_single():
    return load_tvt_n_per_field(1)

def load_tvt_double():
    return load_tvt_n_per_field(2)



def load_tvt_n_per_field_centercrop(n_per):
    '''
    array ims:
      - with dtype float32
      - values in the range 0 to 1
      - shapes like (50000, 28, 28, 1)
    array pos:
      - with dtype float32
      - with values in the range 0 to 28    (i_center, j_center, h, w)
      - shapes like (50000, 5, 4)           for train and n = 5
    array class:
      - with dtype uint8
      - with values in the range 0 to 1
      - shapes like (50000, 5, )            for train and n = 5
    '''

    raw_train_x, raw_train_y, raw_test_x, raw_test_y = mnist.load_data()
    
    train_ims, train_pos, train_class = generate_dataset(raw_train_x, raw_train_y, n_per, 0, 
                            canvas_size = (120,120), center_crop=True, crop_size=(64,64))
    test_ims,  test_pos,  test_class  = generate_dataset(raw_test_x,  raw_test_y,  n_per, 1,
                            canvas_size = (120,120), center_crop=True, crop_size=(64,64))

    return train_ims, train_pos, train_class, test_ims, test_pos, test_class
