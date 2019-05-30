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

import tensorflow as tf
from tf_plus import BatchNormalization, Lambda   # BN + Lambda layers are custom, rest are just from tf.layers
from tf_plus import Conv2D, MaxPooling2D, Flatten, Dense
from tf_plus import he_normal, relu
from tf_plus import Layers, SequentialNetwork, l2reg, PreprocessingLayers
from general.tfutil import tf_reshape_like
from CoordConv import AddCoords, CoordConv

ReLu = Lambda(lambda xx: relu(xx))

class RegionProposalSampler(Layers):
    '''region proposal net rewritten '''

    def __init__(self, rpn_params, bsamp_params, nms_params, l2=0, im_h=64, im_w=64, 
                 coordconv=False, clip=True, filtersame=False):
        super(RegionProposalSampler, self).__init__()
        self.rpn_params = rpn_params
        self.bsamp_params = bsamp_params
        self.nms_params = nms_params
        self.im_h = im_h
        self.im_w = im_w
        self.clip = clip

        _pad = 'same' if filtersame else 'valid'
        _dim = 16 if filtersame else 13
        if coordconv:
            self.l('bottom_conv', SequentialNetwork([
                    AddCoords(x_dim=im_w, y_dim=im_h, with_r=False, skiptile=True), # (batch, 64, 64, 4 or 5)
                    Conv2D(32, (5,5), padding=_pad,
                        kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                    ReLu,
                    MaxPooling2D(pool_size=2, strides=2),
                    Conv2D(64, (5,5), padding=_pad,
                        kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                    ReLu,
                    MaxPooling2D(pool_size=2, strides=2),
                    ], name='bottom_conv'))
            
            self.l('another_conv', SequentialNetwork([
                    AddCoords(x_dim=_dim, y_dim=_dim, with_r=False, skiptile=True), 
                    Conv2D(rpn_params.rpn_hidden_dim, (3,3), padding='same',
                        kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                    ReLu
                    ], name='another_conv'))

            self.l('box_mover', SequentialNetwork([
                    Conv2D(rpn_params.rpn_hidden_dim, (3,3), padding='same',
                        kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                    ReLu,
                    AddCoords(x_dim=_dim, y_dim=_dim, with_r=False, skiptile=True), 
                    Conv2D(4*rpn_params.num_anchors, (1,1), 
                        kernel_initializer=tf.zeros_initializer,
                        bias_initializer=tf.constant_initializer([0.]),
                        kernel_regularizer=l2reg(l2))
                    ], name='box_mover'))  # (13,13,4*k)

        else:
            self.l('bottom_conv', SequentialNetwork([
                    Conv2D(32, (5,5), padding=_pad,
                        kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                    ReLu,
                    MaxPooling2D(pool_size=2, strides=2),
                    Conv2D(64, (5,5), padding=_pad,
                        kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                    ReLu,
                    MaxPooling2D(pool_size=2, strides=2),
                    ], name='bottom_conv'))

            self.l('another_conv', SequentialNetwork([
                    Conv2D(rpn_params.rpn_hidden_dim, (3,3), padding='same',
                        kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                    ReLu
                    ], name='another_conv'))
    
            self.l('box_mover', SequentialNetwork([
                    Conv2D(rpn_params.rpn_hidden_dim, (3,3), padding='same',
                        kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                    ReLu,
                    Conv2D(4*rpn_params.num_anchors, (1,1), 
                        kernel_initializer=tf.zeros_initializer,
                        bias_initializer=tf.constant_initializer([0.]),
                        kernel_regularizer=l2reg(l2))
                    ], name='box_mover'))  # (13,13,4*k)

        self.l('box_scorer', SequentialNetwork([
                Conv2D(2*rpn_params.num_anchors, (1,1), 
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                ], name='box_scorer'))  # (13,13,2*k)

        return 
    
    def call(self, inputs):
    
        input_images, input_anchors, input_gtbox = inputs[0], inputs[1], inputs[2] 
       
        num_anchors = self.rpn_params.num_anchors # K

        img_features = self.bottom_conv(input_images)

        rpn_features = self.another_conv(img_features)
        
        box_mover_logits = self.box_mover(rpn_features)  # (1, 13, 13, 4*K)
        box_scorer_logits = self.box_scorer(rpn_features)   # (1, 13, 13, 2*K)
      
        box_mover_logits_reshaped = reshape_box_features(box_mover_logits, num_anchors, 4)  # (N, 4)
        box_scorer_logits_reshaped = reshape_box_features(box_scorer_logits, num_anchors, 2) # (N, 2)

        anchors_reshaped, edge_mask = reshape_box_features(input_anchors, num_anchors, 4, ret_edge_mask=True) #(num_anchors, 4)
        self.a('edge_mask', edge_mask)

        box_moved = box_transform(anchors_reshaped, box_mover_logits_reshaped) 
        
        if self.clip:
            box_moved, valid_mask = clip_boxes(box_moved, (0, 0, self.im_h-1, self.im_w-1))
            input_gtbox, _ = clip_boxes(input_gtbox, (0, 0, self.im_h-1, self.im_w-1))
        else:
            valid_mask = tf.ones(tf.shape(box_moved)[0], dtype='bool')
        self.a('valid_mask', valid_mask)
        
        self.a('img_features', img_features)
        self.a('rpn_features', rpn_features)
        self.a('box_mover_logits', box_mover_logits)
        self.a('box_mover_logits_reshaped', box_mover_logits_reshaped)
        self.a('box_scorer_logits', box_scorer_logits)
        self.a('box_scorer_logits_reshaped', box_scorer_logits_reshaped)
        self.a('box_moved', box_moved)
        self.a('anchors_reshaped', anchors_reshaped)

        # 'training branch': boxes are sampled w.r.t gt
        iou_matrix = box_iou(box_moved, input_gtbox)
    
        self.a('box_sampler', BoxSampler(self.bsamp_params, self.im_h, self.im_w))
        pos_neg_labels, idx_of_box, idx_of_gt = self.box_sampler([iou_matrix, box_moved])

        self.a('iou_matrix', iou_matrix)            # e.g.
        self.a('pos_neg_labels', pos_neg_labels)    # e.g.
        self.a('idx_of_box', idx_of_box)            # e.g.
        self.a('idx_of_gt', idx_of_gt)              # e.g.
        
        # 'test branch': boxes are sampled by nms 
        self.a('nms_sampler', BoxNMS(self.nms_params.nms_thresh, self.nms_params.max_proposals))
        nms_boxes, nms_scores = self.nms_sampler([self.box_scorer_logits_reshaped, box_moved, valid_mask])

        nms_iou_matrix = box_iou(nms_boxes, input_gtbox) 
        nms_iou_self_matrix = box_iou(nms_boxes, nms_boxes) 
        self.a('nms_boxes', nms_boxes)
        self.a('nms_scores', nms_scores)
        self.a('nms_iou_matrix', nms_iou_matrix)
        self.a('nms_iou_self_matrix', nms_iou_self_matrix)

        self.make_losses_and_metrics()
    
        return nms_boxes
        
    def make_losses_and_metrics(self):

        #box_out = self.box_moved
        box_out = self.box_moved
        score_out = self.box_scorer_logits_reshaped
        trans_out = self.box_mover_logits_reshaped
        anchor_out = self.anchors_reshaped

        ##################
        # make box loss
        ##################
        # select pos box coords from box_out
        pos_box_index = self.box_sampler.pos_indx_samples                  # (psize, 1)
        neg_box_index = self.box_sampler.neg_indx_samples                  # 
        self.a('pos_box_index', pos_box_index)
        self.a('neg_box_index', neg_box_index)
        #pos_box_index = tf.squeeze(pos_box_index, 1)                         # (p_size,)
        pos_box = tf.gather(box_out, pos_box_index)                  # (p_size, 4)
        self.a('pos_box', pos_box)
        pos_score = tf.gather(score_out[:,0], pos_box_index)
        self.a('pos_score', pos_score)

        pos_iou = tf.gather_nd(self.iou_matrix, self.box_sampler.pos_box_gt_indices)
        self.a('pos_box_iou_train', tf.reduce_mean(pos_iou), trackable=True)

        neg_box = tf.gather(box_out, neg_box_index)                  # (p_size, 4)
        self.a('neg_box', neg_box)
        neg_score = tf.gather(score_out[:,1], neg_box_index)
        self.a('neg_score', neg_score)

        # select transform params from trans_out
        trans_select = tf.gather(trans_out, pos_box_index)                  # (p_size, 4)

        # select anchor coords from anchor_out
        anchor_select = tf.gather(anchor_out, pos_box_index)                # (p_size, 4)
        self.a('anchor_select', anchor_select)
        
        # select target coords from gt
        target_index = self.idx_of_gt                           # (p_size, 1)
        #target_index = tf.squeeze(target_index, 1)                           # (p_size,)
        #target_select = tf.gather(self.input_gtbox, target_index)   # (p_size, 4)
        target_select = tf.gather(self.input_gtbox, target_index)   # (p_size, 4)
        self.a('target_index', target_index)
        self.a('target_select', target_select)
       
        # make inverse transform to produce target transform params
        
        target_trans_select = invert_box_transform(anchor_select, target_select)
        
        box_regr_loss = smooth_l1(trans_select - target_trans_select)
        box_loss = tf.reduce_mean(box_regr_loss)

        self.a('box_loss', box_loss, trackable=True)


        ##################
        # make score loss
        ##################
        p_size = self.box_sampler.p_size
        n_size = self.box_sampler.n_size
        self.a('p_size', p_size)
        self.a('n_size', n_size)

        pn_box_index = self.idx_of_box
        pn_scores = tf.gather(score_out, pn_box_index)
        self.a('score_logits', pn_scores)
        
        norm_score = tf.nn.softmax(pn_scores)
        self.a('score_softmax', norm_score)
        
        score_loss = tf.reduce_mean( 
                        tf.nn.softmax_cross_entropy_with_logits_v2(
                            labels=self.pos_neg_labels, logits=pn_scores))
        self.a('score_loss', score_loss, trackable=True)
        
        # regularizers
        reg_loss = tf.losses.get_regularization_loss()
        reg_losses = tf.losses.get_regularization_losses()
        
        self.a('reg_loss', reg_loss, trackable=True)
        
        self.a('loss', box_loss + score_loss + reg_loss, trackable=True)

        #self.a('iou', iou, trackable=True)

        ##################
        # make nms metric
        ##################
        
   

        self.a('mean_nms_iou', tf.reduce_mean(self.nms_iou_matrix), trackable=True)
        self.a('argmax_nms_iou', tf.reduce_max(self.nms_iou_matrix, axis=-1))
        self.a('mean_argmax_nms_iou', tf.reduce_mean(tf.reduce_max(self.nms_iou_matrix, axis=-1)), trackable=True)
        self.a('mean_nms_scores', tf.reduce_mean(self.nms_scores), trackable=True)

        return


def reshape_box_features(box_logits, N, K, ret_edge_mask=False):
    '''
    a conv net outputs box features in (batch, h, w, N*K)
        where N: number of anchors
              K: 4 (box mover) or 2 (box scorer)

    if ret_edge_mask, return a mask of the same length as xx where the positions along h w edges are marked true
        e.g. for h=13, w=13, 
            imat = np.outer(np.arange(13), np.ones(13))
            jmat = np.outer(np.ones(13),np.arange(13))
            jedge = np.where(jmat.reshape(-1) == 12)[0].tolist() + np.where(jmat.reshape(-1) == 0)[0].tolist()
            iedge = np.where(imat.reshape(-1) == 12)[0].tolist() + np.where(imat.reshape(-1) == 0)[0].tolist()
            ijedge = np.unique(iedge+jedge) 

    '''
    # First break N K
    shape_list = box_logits.get_shape().as_list()
    if len(shape_list) == 3:
        h, w = shape_list[0], shape_list[1]
    elif len(shape_list) == 4:
        if shape_list[0] is None:
            shape_list[0] = -1
        h, w = shape_list[1], shape_list[2]
    else:
        raise ValueError('Tensor shape either (batch, h, w, *) or (h, w, *)')

    xx = tf.reshape(box_logits, shape_list[:-1] + [N, K])   # (batch, h, w, N, K)
    # Then reshape
    xx = tf.reshape(xx, [-1, K])

    if ret_edge_mask:

        j_edge0 = tf.range(0,h*w,delta=h)
        j_edge1 = tf.range(h-1,h*w,delta=h)
        i_edge0 = tf.range(h,)
        i_edge1 = tf.range(h*(w-1), h*w)

        edge_idx = tf.unique(tf.concat([j_edge0, j_edge1, i_edge0, i_edge1], axis=0))

        edge_mask_hw = tf.sparse_to_dense(edge_idx, [h*w], 1, default_value=0, validate_indices=False)
        # an edge mask of (h*w,)

        # stack K times
        #(h*w, ) --> (h*w, K)
        edge_mask_hwn = tf.expand_dims(edge_mask_hw, 1) * tf.ones([1,N], tf.int32)
        return xx, tf.reshape(edge_mask_hwn, [-1])
    
    else:
        return xx


def box_transform(in_boxes, transform):
    '''
    in_boxes: orignial box location, e.g. anchors
    transform: transfomrmation parameters
    both are of size (batch, N, 4)
    last dimension of 'in_boxes': ic, jc, h, w
    last dimension of 'transform': ti, tj, th, tw
    '''
    # i_out = i_in + ti * h
    out_boxes_i = in_boxes[...,0] + transform[...,0] * in_boxes[...,2]

    # j_out = j_in + tj * w
    out_boxes_j = in_boxes[...,1] + transform[...,1] * in_boxes[...,3]
    
    # h_out = h_in * exp(th)
    out_boxes_h = in_boxes[...,2] * tf.exp(transform[...,2])
    
    # w_out = w_in * exp(tw)
    out_boxes_w = in_boxes[...,3] * tf.exp(transform[...,3])

    out_boxes = tf.stack([out_boxes_i,out_boxes_j,out_boxes_h,out_boxes_w])  # (4, N)
    out_boxes = tf.transpose(out_boxes)  # (N,4)
    
    return out_boxes


def invert_box_transform(box_before, box_after):
    # box_before: like the anchor
    # box_after: like the target box
    # Both of shape (N, 4) -- 2nd dim: (i, j, h, w)

    # output: transform paramters in to (N, 4) -- 3rd dim: (ti, tj, th, tw)
    # ti = (i_b - i_a) / ha
    out_ti = (box_after[:,0] - box_before[:,0]) / box_before[:,2]
    # tj = (j_b - j_a) / wa
    out_tj = (box_after[:,1] - box_before[:,1]) / box_before[:,3]
    # th = log(h_b / h_a)
    out_th = tf.log(box_after[:,2] / box_before[:,2])
    # w_out = w_in * exp(tw)
    # tw = log(w_b / w_a)
    out_tw = tf.log(box_after[:,3] / box_before[:,3])

    out_transform = tf.stack([out_ti,out_tj,out_th,out_tw])  # (4,n)
    out_transform = tf.transpose(out_transform)  # (n,4)
    return out_transform

def box_iou(box1, box2):
    if len(box1.get_shape()) == 3:
        box1 = tf.squeeze(box1, axis=0)
    if len(box2.get_shape()) == 3:
        box2 = tf.squeeze(box2, axis=0)

    B1 = tf.shape(box1)[0]
    B2 = tf.shape(box2)[0]  

    # convert to low/high points format
    box1_lohi = convert_box_ijhw2ijij(box1)                     # (B1, 4)             -
    box1_lohi_expand = tf.expand_dims(box1_lohi, 1)             # (B1, 1, 4)
    box1_lohi_tiled = tf.tile(box1_lohi_expand, (1,B2,1))       # (B1, B2, 4)
    
    box2_lohi = convert_box_ijhw2ijij(box2)                     # (B2, 4)
    box2_lohi_expand = tf.expand_dims(box2_lohi, 0)             # (1, B2, 4)
    box2_lohi_tiled = tf.tile(box2_lohi_expand, (B1,1,1))       # (B1, B2, 4)

    # compute area of box1 and box2
    area1 = box1[:,2] * box1[:,3]                               # (B1, )
    area1_expand = tf.expand_dims(area1, -1)                    # (B1, 1)
    area1_tiled = tf.tile(area1_expand, (1,B2))                 # (B1, B2)

    area2 = box2[:,2] * box2[:,3]                               # (B2, )
    area2_expand = tf.expand_dims(area2, 0)                     # (1, B2)
    area2_tiled = tf.tile(area2_expand, (B1,1))                        # (B1, B2)

    # compute intersection area coordinates
    # (i0, j0) for upper-left, (i1, j1) for lower-right
    i0 = tf.maximum(box1_lohi_tiled[:,:,0],
                    box2_lohi_tiled[:,:,0])
    j0 = tf.maximum(box1_lohi_tiled[:,:,1],
                   box2_lohi_tiled[:,:,1])
    i1 = tf.minimum(box1_lohi_tiled[:,:,2],
                   box2_lohi_tiled[:,:,2])
    j1 = tf.minimum(box1_lohi_tiled[:,:,3],
                   box2_lohi_tiled[:,:,3])

    h = tf.maximum(i1-i0, 0.)
    w = tf.maximum(j1-j0, 0.)

    intersection = w * h

    # compute intersection over the union.
    iou = intersection / (area1_tiled + area2_tiled - intersection)  # (B1, B2)
    return iou


def clip_boxes(boxes, bounds):
    ''' clip bounding boxes to a specified region
        Inputs:
            - boxes: tensor containing boxes, of shape (N, 4). Only support format icjchw now.
            - bounds: tuple (i_min, j_min, i_max, j_max) containing 4 bounds (inclusive)
        Outputs:
            - boxes_clipped: tensor giving coordinates of clipped boxes, same shape as input boxes
            - valid: 1D byte tensor indicating which bounding boxes are valid, in sense of completely out of the image
    '''
    boxes_hilo = convert_box_ijhw2ijij(boxes)

    i_min, j_min, i_max, j_max = bounds[0], bounds[1], bounds[2], bounds[3]
    boxes_clipped_i0 = tf.clip_by_value(boxes_hilo[:,0], i_min, i_max-1)
    boxes_clipped_j0 = tf.clip_by_value(boxes_hilo[:,1], j_min, j_max-1)
    boxes_clipped_i1 = tf.clip_by_value(boxes_hilo[:,2], i_min+1, i_max)
    boxes_clipped_j1 = tf.clip_by_value(boxes_hilo[:,3], j_min+1, j_max)

    validi = tf.greater(boxes_clipped_i1, boxes_clipped_i0)         # valid i: i1>i0
    validj = tf.greater(boxes_clipped_j1, boxes_clipped_j0)         # valid j: j1>j0

    valid = tf.logical_and(validi,validj)                           # (N,), bool

    boxes_clipped_hilo = tf.stack([boxes_clipped_i0, boxes_clipped_j0, boxes_clipped_i1, boxes_clipped_j1])
    boxes_clipped_hilo = tf.transpose(boxes_clipped_hilo, perm=[1, 0])
    box_clipped = convert_box_ijij2ijhw(boxes_clipped_hilo)

    return box_clipped, valid


def convert_box_ijhw2ijij(box):
    """ Convert box from (ic, jc, h, w) to (i0, j0, i1, j1)
        box shape: (num_of_boxes, 4) 
    """
    halfh = box[:,2] / 2.
    halfw = box[:,3] / 2.
    # i0 = ic - h/2
    out_i0 = box[:,0] - halfh
    # j0 = jc - w/2
    out_j0 = box[:,1] - halfw
    # i1 = ic + h/2
    out_i1 = box[:,0] + halfh
    # j1 = jc + w/2
    out_j1 = box[:,1] + halfw
    out_box = tf.stack([out_i0, out_j0, out_i1, out_j1])    # (4, B)
    return tf.transpose(out_box, perm=[1, 0])               # (B, 4)

def convert_box_ijij2ijhw(box):
    """ Convert box from (i0, j0, i1, j1) to (ic, jc, h, w)
        box shape: (num_of_boxes, 4)  
    """
    i0 = box[:,0]
    i1 = box[:,2]
    j0 = box[:,1]
    j1 = box[:,3]

    ic = (i0 + i1)/2.
    jc = (j0 + j1)/2.
    h = i1 - i0
    w = j1 - j0

    out_box = tf.stack([ic, jc, h, w])                # (4, B)
    return tf.transpose(out_box, perm=[1, 0])               # (B, 4)

class BoxSampler(Layers):
    """ rewritten from ../../keras_ext/region_layers.py BoxSamplerPosNeg class """
    def __init__(self, bsamp_params, im_h, im_w):
        super(BoxSampler, self).__init__()
        self.sample_size = bsamp_params.sample_size
        self.high_thresh = bsamp_params.hi_thresh
        self.low_thresh = bsamp_params.lo_thresh
        self.im_h = im_h
        self.im_w = im_w

    def call(self, inputs):
        iou, box = inputs[0], inputs[1]
        
        p_size = int(self.sample_size / 2)
        n_size = self.sample_size - p_size
        
        B1, B2 = tf.shape(iou)[0], tf.shape(iou)[1]

        # high mask
        high_mask = (iou > self.high_thresh)            # (B1, B2)
        # low mask
        low_mask = (iou < self.low_thresh)              # (B1, B2), bool

        # find boxes that fall outside boundaries, exclude them from high and low masks
        # convert from (ic, jc, h, w) to (i0, j0, i1, j1) for easier bound comparison
        boxes_hilo = convert_box_ijhw2ijij(box)   # (B1, 4)

        # remove out-of-bound boxes

        i_min = 0.
        j_min = 0.
        i_max = tf.to_float(self.im_h) - 1.0
        j_max = tf.to_float(self.im_w) - 1.0

        # 4 outbound masks
        i_min_mask = tf.less(boxes_hilo[:,0], i_min)       # (B1,), bool
        j_min_mask = tf.less(boxes_hilo[:,1], j_min)       # (B1,), bool
        i_max_mask = tf.greater(boxes_hilo[:,2], i_max)       # (B1,), bool
        j_max_mask = tf.greater(boxes_hilo[:,3], j_max)       # (B1,), bool

        # True is invalid
        outbound_mask = tf.logical_or(i_min_mask, j_min_mask)
        outbound_mask = tf.logical_or(outbound_mask, i_max_mask)
        outbound_mask = tf.logical_or(outbound_mask, j_max_mask)

        # flip it so False is invalid
        outbound_mask = tf.logical_not(outbound_mask)           # (B1,), bool
        outbound_mask_r2 = tf.expand_dims(outbound_mask,1)
        #b2 = tf.to_int32(tf.shape(iou)[1])  # somehow B2 is None, ad hoc fix?
        outbound_mask_tiled = tf.tile(outbound_mask_r2,[1,B2])  # (B1, B2), bool, False is invalid
        #outbound_mask_tiled.set_shape((B1,None))
        self.a('outbound_mask', outbound_mask)

        closest_pred_box = tf.argmax(iou, 0)            # (B2)
        closest_mask = tf.transpose(tf.one_hot(closest_pred_box, B1, on_value=True, off_value=False))

        self.a('closest_pred_box', closest_pred_box)
        self.a('closest_mask', closest_mask)

        # positive candidates: (closest) or (high and excluding outbounds)
        #pos_candi_reduce = tf.reduce_any(pos_candi_mask, 1)             # (B1,), bool
        #pos_candi_indx = tf.where(pos_candi_reduce)                     # (some_num,1), int64
        pos_mask1 = tf.logical_and(high_mask, outbound_mask_tiled)
        pos_candi_mask = tf.logical_or(closest_mask, pos_mask1)          # (B1, B2), bool
        pos_candi_indx = tf.where(pos_candi_mask)                        # (some_num, 2)

        self.a('pos_mask1', pos_mask1)
        self.a('pos_candi_mask', pos_candi_mask)
        self.a('pos_candi_indx', pos_candi_indx)


        # create a non_pos mask and indx for later use
        nonpos_mask = tf.logical_not(tf.reduce_any(pos_candi_mask, 1))      # (B1,), bool
        nonpos_candi_indx = tf.where(nonpos_mask)                        # (some_num_neg, 2), int64

        # target index for positive candidates
        ####tar_index_all = tf.argmax(iou, 1)                               # (B1,)
        ####tar_candi_indx = tf.gather(tar_index_all, pos_candi_indx)       # (some_num, 1), int64

        # negative candidates: (not closest) and (low and excluding outbounds)
        neg_mask1 = tf.logical_and(low_mask, outbound_mask_tiled)
        neg_candi_mask = tf.logical_and(tf.logical_not(closest_mask), neg_mask1)
        neg_candi_reduce = tf.reduce_all(neg_candi_mask, 1)                 # (B1)
        neg_candi_indx = tf.where(neg_candi_reduce)                         # (some_num_other_neg)


        # sample from candidates
        # sample p_size pos_candi_indx by shuffling its index space
        n_pos = tf.shape(pos_candi_indx)[0]  # n_pos should always >= B2 because of closest
        #indd = tf.linspace(0., tf.cast(n_pos-1,tf.float32),tf.cast(n_pos, tf.int32))
        indd_possible_pos = tf.range(n_pos)
        indd_shuffle_pos = tf.random_shuffle(indd_possible_pos)
        n_pos_samples = tf.minimum(n_pos, p_size)  # e.g. 128 or lower if n_pos < 128
        self.a('p_size', n_pos_samples)
        indd_samples_pos = tf.slice(indd_shuffle_pos, [0], [n_pos_samples])  #
        #indd_samples = tf.cond(n_pos > p_size,
        #                       lambda: indd_shuffle[:p_size],
        #                       lambda: indd)            # if not enough just take all

        #pos_indx_samples = tf.gather(pos_candi_indx, tf.cast(indd_samples,tf.int32))  # (actual p_size, 1)
        pos_indx_samples = tf.gather(pos_candi_indx, indd_samples_pos)  # (actual p_size, 2)
        #tar_indx_samples = tf.gather(tar_candi_indx, tf.cast(indd_samples,tf.int32))  # (actual p_size, 1)

        # sample n_size neg_candi_indx by shuffling its index space
        n_neg = tf.shape(neg_candi_indx)[0]
        n_nonpos = tf.shape(nonpos_candi_indx)[0]

        # neg sampling 1st condition: if there are neg candidates at all
        #   if there are no nges found from the threshold, take non-positives
        [neg_candi_indx,indd_possible_neg, n_neg_new] = tf.cond(n_neg > 0,
                                lambda: [neg_candi_indx, tf.range(n_neg), n_neg],
                                lambda: [nonpos_candi_indx, tf.range(n_nonpos), n_nonpos]
                                )


        # 2nd condition: if neg candidates are enough to fill out the minibatch
        #   if not, sample with replacement
        indd_shuffle_neg = tf.random_shuffle(indd_possible_neg)

        #class_logits = tf.log(tf.tile([1./n_neg],n_neg))
        #class_logits = tf.ones([1,n_neg])

        indd_samples_neg = tf.cond(
            n_neg_new > n_size,
            lambda: indd_shuffle_neg[:n_size],
            lambda: tf.random_uniform([tf.zeros([],'int64')], maxval=tf.ones_like((n_neg_new)), dtype=indd_shuffle_neg.dtype)
        )

        neg_indx_samples = tf.squeeze(tf.gather(neg_candi_indx, indd_samples_neg))
        self.a('n_size', tf.shape(neg_indx_samples)[0])

        pos_indx_samples_col1 = pos_indx_samples[:,0]
        pos_indx_samples_col2 = pos_indx_samples[:,1]
        pos_labels = tf.stack([tf.ones_like(pos_indx_samples_col1), tf.zeros_like(pos_indx_samples_col1)], 1)
        neg_labels = tf.stack([tf.zeros_like(neg_indx_samples), tf.ones_like(neg_indx_samples)], 1)
        pos_neg_labels = tf.concat([pos_labels, neg_labels], 0) 

        idx_of_box = tf.concat([pos_indx_samples_col1, neg_indx_samples], 0)
        #idx_of_gt = tf.concat([pos_indx_samples_col2, -1*tf.ones_like(neg_indx_samples)], 0)
        idx_of_gt = pos_indx_samples_col2
        
        self.a('pos_indx_samples', pos_indx_samples_col1)
        self.a('pos_box_gt_indices', pos_indx_samples)
        self.a('neg_indx_samples', neg_indx_samples)

        #self.a('idx_of_box', idx_of_box)
        #self.a('idx_of_gt', idx_of_gt)

        #output = tf.concat([output_c1, output_c2, output_c3], 1)
        return pos_neg_labels, idx_of_box, idx_of_gt


class BoxNMS(Layers):
    """ rewritten from ../../keras_ext/region_layers.py BoxNMS class """
    def __init__(self, nms_thresh, max_proposals):
        super(BoxNMS, self).__init__()
        self.nms_thresh = nms_thresh
        self.max_proposals = max_proposals

    def call(self, inputs):
        scores, boxes, valid_mask = inputs[0], inputs[1], inputs[2]
        # scores: (N, 2)
        # boxes: (N, 4)
        # valid_mask: (N, 1)

        # convert objectnetss negative / positive scores to probabilities
        probs = tf.nn.softmax(scores)                       # (N, 2)
        scores = probs[:,0]                                     # (N,)  probability positive
        scores = tf.boolean_mask(scores, valid_mask)
        boxes = tf.boolean_mask(boxes, valid_mask)

        #pick_indx = self._nms(scores, boxes, self.nms_thresh, self.max_proposals)
        boxes_hilo = convert_box_ijhw2ijij(boxes)
        selected_indices = tf.image.non_max_suppression(boxes_hilo, scores, self.max_proposals, 
                                            iou_threshold=self.nms_thresh, name='tf_nms')
        nms_boxes = tf.gather(boxes, selected_indices)
        nms_scores = tf.gather(scores, selected_indices)

        return nms_boxes, nms_scores 


def smooth_l1(x):
    absx = tf.abs(x)
    big = tf.cast(tf.greater(absx, 1), tf.float32)
    loss4 = tf.multiply(big, absx) + tf.multiply((1-big), tf.square(x))
    loss = tf.reduce_sum(loss4, 1)
    return loss
    


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

class AddCoords(Layers):
        """A more elegant layer than CoordsUnroll"""
        def __init__(self, x_dim=64, y_dim=64, with_r=False, skiptile=True, skipconcat=False):
            super(AddCoords, self).__init__()
            self.x_dim = x_dim
            self.y_dim = y_dim
            self.with_r = with_r
            self.skiptile = skiptile
            self.skipconcat = skipconcat

        def call(self, input_tensor):
            """
            input_tensor: eg. (batch, 1, 1, c), or (batch, x_dim, y_dim, c)
            In the first case, first tile the input_tensor to be (batch, x_dim, y_dim, c)
            In the second case, skiptile, just concat
            """
            if not self.skiptile:
                input_tensor = tf.tile(input_tensor, [1, self.x_dim, self.y_dim, 1]) # (batch, 64, 64, 2)

            batch_size_tensor = tf.shape(input_tensor)[0]  # get batch size
            
            xx_ones = tf.ones([batch_size_tensor, self.x_dim], dtype=tf.int32)  # (batch, 64)
            xx_ones = tf.expand_dims(xx_ones, -1)                               # (batch, 64, 1)
            xx_range = tf.tile(tf.expand_dims(tf.range(self.x_dim), 0), [batch_size_tensor, 1]) # (batch, 64)
            xx_range = tf.expand_dims(xx_range, 1)                              # (batch, 1, 64)

            xx_channel = tf.matmul(xx_ones, xx_range)                           # (batch, 64, 64)
            xx_channel = tf.expand_dims(xx_channel, -1)                         # (batch, 64, 64, 1)

            yy_ones = tf.ones([batch_size_tensor, self.y_dim], dtype=tf.int32)  # (batch, 64)
            yy_ones = tf.expand_dims(yy_ones, 1)                                # (batch, 1, 64)
            yy_range = tf.tile(tf.expand_dims(tf.range(self.y_dim), 0), [batch_size_tensor, 1]) # (batch, 64)
            yy_range = tf.expand_dims(yy_range, -1)                             # (batch, 64, 1)

            yy_channel = tf.matmul(yy_range, yy_ones)                           # (batch, 64, 64)
            yy_channel = tf.expand_dims(yy_channel, -1)                         # (batch, 64, 64, 1)

            # cast to float and normalize coords
            input_tensor = tf.cast(input_tensor, 'float32')

            xx_channel = tf.cast(xx_channel, 'float32') / (self.x_dim - 1)
            yy_channel = tf.cast(yy_channel, 'float32') / (self.y_dim - 1)
            
            if self.skipconcat:
                ret = tf.concat([xx_channel, yy_channel], axis=-1)              # (batch, 64, 64, 2) 
            else:
                ret = tf.concat([input_tensor, xx_channel, yy_channel], axis=-1)    # (batch, 64, 64, c+2)

            if self.with_r:
                rr = tf.sqrt( tf.square(xx_channel-0.5)
                            + tf.square(yy_channel-0.5)
                            )                
                ret = tf.concat([ret, rr], axis=-1)

            return ret


