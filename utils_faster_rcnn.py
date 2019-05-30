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

import skimage
import skimage.io
import skimage.transform
import numpy as np
from matplotlib.pyplot import *
from IPython import embed

from general.conv_calc import receptive_calc_conv, receptive_calc_conv_same


def plot_boxes(im, pos, box_coords=None, obj_scores=None):
    assert len(im.shape) == 3, 'only supports one img plotting'
    gray()
    subplot(111)
    imshow(im.squeeze())
    # show true box
    for i in range(pos.shape[0]):
        i_center, j_center, height, width = pos[i]
        gca().add_patch(Rectangle((j_center - width/2, i_center - height/2), width, height, ec='m', lw=2.5, fc=(0,0,0,0)))
    
    # show output boxes
    if box_coords is not None:
        for i in range(box_coords.shape[0]):
            bi_c, bj_c, bh, bw = box_coords[i]
            gca().add_patch(Rectangle((bj_c - bw/2, bi_c - bh/2), bw, bh, ec='g', fc=(0,0,0,0)))
            if obj_scores is not None:
                text(bj_c - bw/2, bi_c - bh/2, '%.2f,%.2f'%(obj_scores[i][0],obj_scores[i][1]), color='green')
    show()
    return

def plot_pos_boxes(im, pos, box_coords, pos_scores, showlabel=True):
    assert len(im.shape) == 3, 'only supports one img plotting'
    gray()
    #subplot(111)
    imshow(im.squeeze())
    # show true box
    for i in range(pos.shape[0]):
        i_center, j_center, height, width = pos[i]
        gca().add_patch(Rectangle((j_center - width/2, i_center - height/2), width, height, ec='m', lw=2.5, fc=(0,0,0,0)))
    
    # show output boxes
    for i in range(box_coords.shape[0]):
        bi_c, bj_c, bh, bw = box_coords[i]
        gca().add_patch(Rectangle((bj_c - bw/2, bi_c - bh/2), bw, bh, ec='g', fc=(0,0,0,0)))
        if showlabel:
            text(bj_c - bw/2, bi_c - bh/2, '%d'%(pos_scores[i]), color='green')
        else:
            text(bj_c - bw/2, bi_c - bh/2, '%.2f'%(pos_scores[i]), color='green')
    #show()
    return

def plot_boxes_pos_neg(im, gt, pos_box, neg_box):
    assert len(im.shape) == 3, 'only supports one img plotting'
    gray()
    #subplot(111)
    imshow(im.squeeze())
    # show true box
    for i in range(gt.shape[0]):
        i_center, j_center, height, width = gt[i]
        gca().add_patch(Rectangle((j_center - width/2, i_center - height/2), width, height, ec='m', lw=2.5, fc=(0,0,0,0)))
    
    # show output boxes
    for i in range(pos_box.shape[0]):
        bi_c, bj_c, bh, bw = pos_box[i]
        gca().add_patch(Rectangle((bj_c - bw/2, bi_c - bh/2), bw, bh, ec='g', lw=1.5, fc=(0,0,0,0)))
    for j in range(neg_box.shape[0]):
        bi_c, bj_c, bh, bw = neg_box[j]
        gca().add_patch(Rectangle((bj_c - bw/2, bi_c - bh/2), bw, bh, ec='r', fc=(0,0,0,0)))

    
    #show()
    return

def plot_pos_boxes_thickness(im, pos, box_coords, pos_scores):
    assert len(im.shape) == 3, 'only supports one img plotting'
    gray()
    #subplot(111)
    imshow(im.squeeze())
    # show true box
    for i in range(pos.shape[0]):
        i_center, j_center, height, width = pos[i]
        gca().add_patch(Rectangle((j_center - width/2, i_center - height/2), width, height, ec='m', lw=2.5, fc=(0,0,0,0)))
    
    # show output boxes
    for i in range(box_coords.shape[0]):
        bi_c, bj_c, bh, bw = box_coords[i]
        gca().add_patch(Rectangle((bj_c - bw/2, bi_c - bh/2), bw, bh, ec='g', lw=pos_scores[i], fc=(0,0,0,0)))
        #if showlabel:
        #    text(bj_c - bw/2, bi_c - bh/2, '%d'%(pos_scores[i]), color='green')
        #else:
        #    text(bj_c - bw/2, bi_c - bh/2, '%.2f'%(pos_scores[i]), color='green')
    #show()
    return

def make_anchors_numpy(featuremap_hw, batch_size, anchor_templates, receptsize_stride_offset):
    # step 1: compute field center  
    #   derived from net_utils.lua in Densecap code
    #   calculates how to map each point on feature map back to original img
    #

    # For simple conv(5).pool(2).conv(5).pool(2) model:    ( 16,  4,   0)
    # For VGG without final pool layer:                    (196, 16, -90)
    recept_size, stride, offset = receptsize_stride_offset

    i0 = j0 = recept_size/2
    si = sj = stride
    # step 2: make anchors in terms of coords on original img, based on current featuremap locations and anchor templates
    # anchor_templates: (k,2) array presenting k anchor box templates in (box height, box width)
    # return anchors: (B, 4k, h, w) numpy array (dtype=float32), if reshape to (B, k, 4, h, w) then along the 3rd dim we have (ic, jc, bh, bw) 
    h, w = featuremap_hw
    i_centers = np.arange(0, float(h))  # i_centers in the feature map
    i_centers *= si
    i_centers += i0 + offset                     # now tranformed to i_centers in original image
    j_centers = np.arange(0, float(w))
    j_centers *= sj
    j_centers += j0 + offset
    
    box_heights = anchor_templates[:,0]
    box_widths = anchor_templates[:,1]

    print('ic',i_centers)
    print('jc',j_centers)
    print('bh',box_heights)
    print('bw',box_widths)

    # empty output
    k = len(anchor_templates)
    anchors = np.empty([4, k, h, w])
   
    anchors[0] = np.tile(i_centers.reshape((1,-1,1)), (k,1,w))        # ic
    anchors[1] = np.tile(j_centers.reshape((1,1,-1)), (k,h,1))        # jc
    anchors[2] = np.tile(box_heights.reshape((-1,1,1)), (1,h,w))      # box_h
    anchors[3] = np.tile(box_widths.reshape((-1,1,1)), (1,h,w))       # box_w
    
    anchors = np.expand_dims(anchors,axis=0)
    anchors = np.tile(anchors,(batch_size,1,1,1,1)) # (B, 4, k, h, w)
    anchors = np.reshape(anchors, (batch_size, 4*k, h, w))   # (B, 4*k, h, w)
    anchors = np.transpose(anchors, (0, 2, 3, 1)) # (B, h, w, 4*k)
    
    return anchors.astype('float32')


def dense_to_one_hot(labels_dense, num_class=10):
    # for more than one digits in a image
    b, n = np.shape(labels_dense)
    index_offset = np.arange(b*n) * num_class
    labels_one_hot = np.zeros((b, n, num_class))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def make_anchors_mnist(featuremap_hw, batch_size, anchor_templates):
    def calc_conv(w,s):
        return lambda rs: (rs[0]+rs[1]*(w-1), rs[1]*s)

    def calc_pool(w,s):
        return lambda rs: (rs[0]+rs[1]*(w-1), rs[1]*s)
    # step 1: compute field center  
    #   derived from net_utils.lua in Densecap code
    #   calculates how to map each point on feature map back to original img
    #   hardcoded (for now) each layer's filter size and stride information
    #
    rs = (1.,1.)
    rs = calc_conv(5,1)(rs) # 1st conv
    rs = calc_pool(2,2)(rs) # 1st pool
    rs = calc_conv(5,1)(rs) # 2nd conv
    rs = calc_pool(2,2)(rs) # 2nd pool
    
    i0 = j0 = rs[0]/2
    si = sj = rs[1]
    # step 2: make anchors in terms of coords on original img, based on current featuremap locations and anchor templates
    # anchor_templates: (k,2) array presenting k anchor box templates in (box height, box width)
    # return anchors: (B, 4k, h, w) numpy array (dtype=float32), if reshape to (B, k, 4, h, w) then along the 3rd dim we have (ic, jc, bh, bw) 
    h, w = featuremap_hw
    i_centers = np.arange(0, float(h))  # i_centers in the feature map
    i_centers *= si
    i_centers += i0                     # now tranformed to i_centers in original image
    j_centers = np.arange(0, float(w))
    j_centers *= sj
    j_centers += j0
    
    box_heights = anchor_templates[:,0]
    box_widths = anchor_templates[:,1]

    print('ic',i_centers)
    print('jc',j_centers)
    print('bh',box_heights)
    print('bw',box_widths)

    # empty output
    k = len(anchor_templates)
    anchors = np.empty([4, k, h, w])
   
    anchors[0] = np.tile(i_centers.reshape((1,-1,1)), (k,1,w))        # ic
    anchors[1] = np.tile(j_centers.reshape((1,1,-1)), (k,h,1))        # jc
    anchors[2] = np.tile(box_heights.reshape((-1,1,1)), (1,h,w))      # box_h
    anchors[3] = np.tile(box_widths.reshape((-1,1,1)), (1,h,w))       # box_w
    
    anchors = np.expand_dims(anchors,axis=0)
    anchors = np.tile(anchors,(batch_size,1,1,1,1)) # (B, 4, k, h, w)
    anchors = np.reshape(anchors, (batch_size, 4*k, h, w))   # (B, 4*k, h, w)
    anchors = np.transpose(anchors, (0, 2, 3, 1)) # (B, h, w, 4*k)

    return anchors.astype('float32')


def make_anchors_mnist_same(featuremap_hw, batch_size, anchor_templates):

    rso = (1.,1., 0.)
    rso = receptive_calc_conv_same(5,1)(rso)
    rso = receptive_calc_conv(2,2)(rso) # 1st pool
    rso = receptive_calc_conv_same(5,1)(rso) # 2nd conv
    rso = receptive_calc_conv(2,2)(rso) # 2nd pool
    
    recept_size, stride, offset = rso 
    i0 = j0 = recept_size/2
    si = sj = stride
    # step 2: make anchors in terms of coords on original img, based on current featuremap locations and anchor templates
    # anchor_templates: (k,2) array presenting k anchor box templates in (box height, box width)
    # return anchors: (B, 4k, h, w) numpy array (dtype=float32), if reshape to (B, k, 4, h, w) then along the 3rd dim we have (ic, jc, bh, bw) 
    h, w = featuremap_hw
    i_centers = np.arange(0, float(h))  # i_centers in the feature map
    i_centers *= si
    i_centers += i0 + offset                 # now tranformed to i_centers in original image
    j_centers = np.arange(0, float(w))
    j_centers *= sj
    j_centers += j0 + offset
    
    box_heights = anchor_templates[:,0]
    box_widths = anchor_templates[:,1]

    print('ic',i_centers)
    print('jc',j_centers)
    print('bh',box_heights)
    print('bw',box_widths)

    # empty output
    k = len(anchor_templates)
    anchors = np.empty([4, k, h, w])
   
    anchors[0] = np.tile(i_centers.reshape((1,-1,1)), (k,1,w))        # ic
    anchors[1] = np.tile(j_centers.reshape((1,1,-1)), (k,h,1))        # jc
    anchors[2] = np.tile(box_heights.reshape((-1,1,1)), (1,h,w))      # box_h
    anchors[3] = np.tile(box_widths.reshape((-1,1,1)), (1,h,w))       # box_w
    
    anchors = np.expand_dims(anchors,axis=0)
    anchors = np.tile(anchors,(batch_size,1,1,1,1)) # (B, 4, k, h, w)
    anchors = np.reshape(anchors, (batch_size, 4*k, h, w))   # (B, 4*k, h, w)
    anchors = np.transpose(anchors, (0, 2, 3, 1)) # (B, h, w, 4*k)

    return anchors.astype('float32')
