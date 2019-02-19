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
from tf_plus import Conv2D, MaxPooling2D, Flatten, Dense, UpSampling2D
from tf_plus import he_normal, relu
from tf_plus import Layers, SequentialNetwork, l2reg
from general.tfutil import tf_reshape_like
from CoordConv import AddCoords, CoordConv

Deconv = tf.layers.Conv2DTranspose
ReLu = Lambda(lambda xx: relu(xx))
LReLu = Lambda(lambda xx: lrelu(xx))
Softmax = Lambda(lambda xx: tf.nn.softmax(xx, axis=-1))
Tanh = Lambda(lambda xx: tf.nn.tanh(xx))
GlobalPooling = Lambda(lambda xx: tf.reduce_mean(xx,[1,2]))

class DeconvPainter(Layers):
    '''A Deconv net that paints an image as directed by x,y coord inputs '''

    def __init__(self, l2=0, x_dim=64, y_dim=64, fs=3, mul=1, 
                 onthefly=True, use_mse_loss=False, use_sigm_loss=False):
        super(DeconvPainter, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.onthefly = onthefly
        self.use_mse_loss = use_mse_loss
        self.use_sigm_loss = use_sigm_loss
        
        with tf.variable_scope("model"):
            self.l('model', SequentialNetwork([
                   Lambda(lambda xx: tf.reshape(xx, [-1, 1, 1, 2])),
                   Deconv(64*mul, (fs,fs), (2,2), padding='same',
                       kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                        # output_shape=[None, 2, 2, 64]
                   BatchNormalization(momentum=0.9, epsilon=1e-5),
                   ReLu,
                   Deconv(64*mul, (fs,fs), (2,2), padding='same',  
                       kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                       # output_shape=[None, 4, 4, 64]
                   BatchNormalization(momentum=0.9, epsilon=1e-5),
                   ReLu,
                   Deconv(64*mul, (fs,fs), (2,2), padding='same',
                       kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                       # output_shape=[None, 8, 8, 64]
                   BatchNormalization(momentum=0.9, epsilon=1e-5),
                   ReLu,
                   Deconv(32*mul, (fs,fs), (2,2), padding='same',
                       kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                       # output_shape=[None, 16, 16, 32]
                   BatchNormalization(momentum=0.9, epsilon=1e-5),
                   ReLu,
                   Deconv(32*mul, (fs,fs), (2,2), padding='same',  
                       kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                       # output_shape=[None, 32, 32, 32]
                   BatchNormalization(momentum=0.9, epsilon=1e-5),
                   ReLu,
                   Deconv(1, (fs,fs), (2,2), padding='same',  
                       kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                       # output_shape=[None, 64, 64, 1]
                   ], name='deconv_painter'))

        return 
    
    def call(self, inputs):
        # Shapes:
        # input_coords: (batch, 2)
        # logits: (batch, x_dim, y_dim, 1)
        # logits_flat: (batch, x_dim*y_dim)
        
        input_coords = inputs[0]
        
        logits = self.model([input_coords])
        logits = tf.identity(logits, name='logits') # just to rename it

        logits_flat = Flatten()(logits)

        batch_size_tensor = tf.shape(input_coords)[0]  # get batch size
        
        prob_flat = tf.nn.softmax(logits_flat)
        prob = tf_reshape_like(prob_flat, logits, name='softmax_prob')
        
        pixelwise_prob_flat = tf.nn.sigmoid(logits_flat)
        pixelwise_prob = tf_reshape_like(pixelwise_prob_flat, logits, name='pixelwise_prob')

        if self.onthefly: # make labels on the fly
            indices = tf.reshape(tf.range(batch_size_tensor), [batch_size_tensor, 1])
            concat_indices = tf.concat([indices, input_coords], 1)
            labels_shape = [batch_size_tensor, self.x_dim, self.y_dim]
            labels = tf.sparse_to_dense(concat_indices, labels_shape, 1.)
            self.a('concat_indices', concat_indices)
        else:
            assert len(inputs) == 2, "Not on-the-fly, supply a target image"
            labels = inputs[1]

        labels_flat = Flatten()(labels)

        self.a('prob', prob)
        self.a('prob_flat', prob_flat)
        self.a('pixelwise_prob', pixelwise_prob)
        self.a('pixelwise_prob_flat', pixelwise_prob_flat)
        self.a('logits', logits)
        self.a('logits_flat', logits_flat)
        self.a('labels', labels)
        self.a('labels_flat', labels_flat)

        self.make_losses_and_metrics()
        return logits

    def make_losses_and_metrics(self):

        xe_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits_flat, labels=self.labels_flat))
      
        mse_loss = tf.reduce_mean(
                    tf.pow(self.logits_flat - self.labels_flat, 2))
        # Validated -- the same as above 
        #mse_loss_tf = tf.losses.mean_squared_error(labels=self.labels_flat, predictions=self.logits_flat)
        #self.a('mse_loss_tf', mse_loss_tf, trackable=True)

        sigm_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits_flat, labels=self.labels_flat))
        
        # regularizers
        reg_loss = tf.losses.get_regularization_loss()
        reg_losses = tf.losses.get_regularization_losses()

        argmax_prob = tf.argmax(self.prob_flat, 1)   # index in [0,64*64)
        # convert indices to 2D coordinates
        argmax_x = argmax_prob // self.x_dim
        argmax_y = argmax_prob % self.x_dim

        argmax_label = tf.argmax(self.labels_flat, 1)

        argmax_x_l = tf.floordiv(argmax_label, self.y_dim)
        argmax_y_l = tf.mod(argmax_label, self.y_dim)

        self.a('argmax_prob', argmax_prob)
        self.a('argmax_label', argmax_label)
        self.a('argmax_x', argmax_x)
        self.a('argmax_y', argmax_y)
        self.a('argmax_x_l', argmax_x_l)
        self.a('argmax_y_l', argmax_y_l)


        correct = tf.equal(argmax_prob, argmax_label)
        accuracy = tf.reduce_mean(tf.to_float(correct))
        eucl_dist = tf.reduce_mean(tf.sqrt(tf.to_float(tf.square(argmax_x-argmax_x_l) + tf.square(argmax_y-argmax_y_l))))
        manh_dist = tf.reduce_mean(tf.to_float(tf.abs(argmax_x-argmax_x_l) + tf.abs(argmax_y-argmax_y_l)))

        self.a('reg_losses', reg_losses)
        self.a('correct', correct)
        self.a('accuracy', accuracy, trackable=True)
        self.a('eucl_dist', eucl_dist, trackable=True)
        self.a('manh_dist', manh_dist, trackable=True)
        self.a('xe_loss', xe_loss, trackable=True)
        self.a('mse_loss', mse_loss, trackable=True)
        self.a('sigm_loss', sigm_loss, trackable=True)
        self.a('reg_loss', reg_loss, trackable=True)

        # calculate IOU, useful if input_label is an image

        painted = tf.round(self.pixelwise_prob_flat)  # 1 if > 0.5, 0 if <= 0.5
        n_painted = tf.reduce_mean(tf.reduce_sum(painted, 1))
        
        n_intersection = tf.reduce_sum(tf.multiply(painted, self.labels_flat), 1) # num of pixels in intersection
        
        n_union = tf.reduce_sum(tf.to_float(tf.logical_or(tf.cast(painted, tf.bool), tf.cast(self.labels_flat, tf.bool))), 1)

        self.a('painted', painted)
        self.a('n_intersection', n_intersection)
        self.a('n_union', n_union)

        intersect_pixs = tf.reduce_mean(n_intersection)
        iou = tf.reduce_mean(tf.div(n_intersection, n_union))

        self.a('painted_pixs', n_painted, trackable=True)
        self.a('intersect_pixs', intersect_pixs, trackable=True)
        self.a('iou', iou, trackable=True)


        if self.use_mse_loss:
            self.a('loss', mse_loss+reg_loss, trackable=True)
        elif self.use_sigm_loss:
            self.a('loss', sigm_loss+reg_loss, trackable=True)
        else:
            self.a('loss', xe_loss+reg_loss, trackable=True)

        return

class ConvImagePainter(Layers):
    '''A model net that paints a circle with one-hot center inputs'''

    def __init__(self, l2=0, fs=3, mul=1, use_sigm_loss=True,
                 use_mse_loss=False, version='working'):
        super(ConvImagePainter, self).__init__()
        self.use_sigm_loss = use_sigm_loss
        self.use_mse_loss = use_mse_loss
        assert version in ['simple', 'working', 'dilation'], "model version not supported"
        self.version = version
        
        if version == 'simple':
            net = build_simple_one_channel_onehot2image(l2, name='model')
            self.l('model', net)

        elif version == 'working':
            net = build_working_conv_onehot2image(l2, mul, fs, name='model')
            self.l('model', net)
        
        return 
    
    def call(self, inputs):
      
        assert len(inputs) == 2, "model requires 2 tensors: input_1hot, input_images"
        input_1hot, labels = inputs[0], inputs[1]
        
        logits = self.model(input_1hot)
        logits = tf.identity(logits, name='logits') # just to rename it

        logits_flat = Flatten()(logits)
       
        # Shapes:
        # logits: (batch, x_dim, y_dim, 1)
        # logits_flat: (batch, x_dim*y_dim)
        
        pixelwise_prob_flat = tf.nn.sigmoid(logits_flat)
        pixelwise_prob = tf_reshape_like(pixelwise_prob_flat, logits, name='pixelwise_prob')

        labels_flat = Flatten()(labels)

        self.a('logits', logits)
        self.a('logits_flat', logits_flat)
        self.a('pixelwise_prob', pixelwise_prob)
        self.a('pixelwise_prob_flat', pixelwise_prob_flat)
        self.a('labels', labels)
        self.a('labels_flat', labels_flat)

        self.make_losses_and_metrics()
        return logits

    def make_losses_and_metrics(self):

        mse_loss = tf.reduce_mean(
                    tf.pow(self.logits_flat - self.labels_flat, 2))

        sigm_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits_flat, labels=self.labels_flat))
        
        # regularizers
        reg_loss = tf.losses.get_regularization_loss()
        reg_losses = tf.losses.get_regularization_losses()

        self.a('mse_loss', mse_loss, trackable=True)
        self.a('sigm_loss', sigm_loss, trackable=True)
        self.a('reg_loss', reg_loss, trackable=True)

        # calculate IOU, useful if input_label is an image

        painted = tf.round(self.pixelwise_prob_flat)  # 1 if > 0.5, 0 if <= 0.5
        n_painted = tf.reduce_mean(tf.reduce_sum(painted, 1))

        n_intersection = tf.reduce_sum(tf.multiply(painted, self.labels_flat), 1) # num of pixels in intersection
        n_union = tf.reduce_sum(
                tf.to_float(tf.logical_or(tf.cast(painted, tf.bool), 
                            tf.cast(self.labels_flat, tf.bool))),
                1)

        self.a('painted', painted)
        self.a('n_intersection', n_intersection)
        self.a('n_union', n_union)

        intersect_pixs = tf.reduce_mean(n_intersection)
        iou = tf.reduce_mean(tf.div(n_intersection, n_union))

        self.a('painted_pixs', n_painted, trackable=True)
        self.a('intersect_pixs', intersect_pixs, trackable=True)
        self.a('iou', iou, trackable=True)


        if self.use_mse_loss:
            self.a('loss', mse_loss+reg_loss, trackable=True)
        elif self.use_sigm_loss:
            self.a('loss', sigm_loss+reg_loss, trackable=True)
        else:
            raise ValueError('use either sigmoid or mse loss')

        return

class ConvRegressor(Layers):
    '''A model net that paints a circle with one-hot center inputs'''

    def __init__(self, l2=0, x_dim=64, y_dim=64, fs=3,
                 mul=1, _type='conv_uniform'):
        self.type=_type

        super(ConvRegressor, self).__init__()
        include_r = False

        def coordconv_model():
        #from onehots to coordinate with coordinate augmentation at beginning

            return SequentialNetwork([
                AddCoords(x_dim=x_dim, y_dim=y_dim, with_r=include_r, skiptile=True), # (batch, 64, 64, 4 or 5)
                Conv2D(8, (1,1), padding='same',
                       kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                Conv2D(8, (1,1), padding='same',
                       kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                Conv2D(8, (1,1), padding='same',
                       kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                Conv2D(8, (fs,fs), padding='same',
                       kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                Conv2D(2, (fs,fs), padding='same',
                        kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                MaxPooling2D(pool_size=64,strides=64,padding='valid'),

                ])

        def big_conv2():

            return SequentialNetwork([
                Conv2D(16*mul, (5,5), padding='same',
                   kernel_initializer=he_normal, strides=2,kernel_regularizer=l2reg(l2)),  
                ReLu,
                Conv2D(16*mul, (1,1), padding='same',strides=1,
                       kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                ReLu,

                BatchNormalization(momentum=0.9, epsilon=1e-5),
                Conv2D(16*mul, (3,3), padding='same',strides=1,
                   kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                ReLu,

                Conv2D(16*mul, (3,3), padding='same',
                   kernel_initializer=he_normal, strides=2,kernel_regularizer=l2reg(l2)),  
                ReLu,
                Conv2D(16*mul, (3,3), padding='same',
                       kernel_initializer=he_normal, strides=2,kernel_regularizer=l2reg(l2)),  
                BatchNormalization(momentum=0.9, epsilon=1e-5),
                ReLu,
                Conv2D(16*mul, (fs,fs), padding='same',strides=2,
                       kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                ReLu,
                Conv2D(16*mul, (1,1), padding='same',strides=1,
                       kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                ReLu,
                Conv2D(16*mul, (fs,fs), padding='same',strides=2,
                       kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                ReLu,
                Conv2D(2, (fs,fs), padding='same',
                       kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                ReLu,
                GlobalPooling #pool_size=-1,strides=1,padding='valid'),
                ])

        def big_conv():

            return SequentialNetwork([
                Conv2D(16*mul, (3,3), padding='same',
                   kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                ReLu,
                MaxPooling2D(pool_size=2,strides=2,padding='valid'),
                Conv2D(16*mul, (3,3), padding='same',
                   kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                ReLu,
                MaxPooling2D(pool_size=2,strides=2,padding='valid'),
                Conv2D(16*mul, (3,3), padding='same',
                       kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                ReLu,
                MaxPooling2D(pool_size=2,strides=2,padding='valid'),
                Conv2D(16*mul, (fs,fs), padding='same',
                       kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                ReLu,
                Flatten(),
                Dense(64),
                ReLu,
                Dense(2) #pool_size=-1,strides=1,padding='valid'),
                ])


        with tf.variable_scope("model"):
            if self.type=='conv_uniform':
                self.l('model',big_conv())
            elif self.type=='conv_quarant':
                self.l('model',big_conv2())
            elif self.type=='coordconv':
                self.l('model',coordconv_model())

        return 

    def call(self, inputs):
      
        assert len(inputs) == 2, "model requires 2 tensors: input_images (onehots), target coordinates"
        input_images, labels = inputs[0], inputs[1]

        logits = self.model(input_images)
        logits_flat = Flatten()(logits)
       
        # Shapes:
        # logits: (batch, x_dim, y_dim, 1)
        # logits_flat: (batch, x_dim*y_dim)
        labels_flat = tf.cast(Flatten()(labels),tf.float32)

        self.a('logits', logits_flat)
        #self.a('logits_flat', logits_flat)
        self.a('labels', labels_flat)
        #self.a('labels_flat', labels_flat)

        self.make_losses_and_metrics()

        return logits

    def make_losses_and_metrics(self):

        mse_loss = tf.reduce_mean(
                    tf.square(self.logits - self.labels))

        sigm_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits, labels=self.labels))
        
        # regularizers
        reg_loss = tf.losses.get_regularization_loss()
        reg_losses = tf.losses.get_regularization_losses()

        eucl_dist = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.to_float(tf.square((64.0*self.logits)-(64.0*self.labels))),-1)))
        self.a('eucl_dist', eucl_dist, trackable=True)
        #self.a('mse_dist', mse_loss, trackable=True)

        self.a('loss', mse_loss+reg_loss, trackable=True)
        return

class CoordConvPainter(Layers):
    '''A CoordConv that paints a pixel as directed by x,y coord inputs'''

    def __init__(self, l2=0, x_dim=64, y_dim=64, mul=1, include_r=False,
                 use_mse_loss=False, use_sigm_loss=False):
        super(CoordConvPainter, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.use_mse_loss = use_mse_loss
        self.use_sigm_loss = use_sigm_loss
        
        with tf.variable_scope("model"):
            self.l('coordconvprep', SequentialNetwork([
                Lambda(lambda xx: tf.reshape(xx, [-1, 1, 1, 2])),
                AddCoords(x_dim=x_dim, y_dim=y_dim, with_r=include_r, skiptile=False), # (batch, 64, 64, 4 or 5)
                ], name='coordconvprep'))
            
            self.l('model', SequentialNetwork([
                Conv2D(32*mul, (1,1), padding='valid',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                ReLu,
                Conv2D(32*mul, (1,1), padding='valid',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                ReLu,
                Conv2D(64*mul, (1,1), padding='valid',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                ReLu,
                Conv2D(64*mul, (1,1), padding='valid',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                ReLu,
                Conv2D(1, (1,1), padding='same',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                ], name='coordconv_painter'))
        return 
    
    def call(self, inputs):
        # Shapes:
        # input_coords: (batch, 2)
        # logits: (batch, x_dim, y_dim, 1)
        # logits_flat: (batch, x_dim*y_dim)
        
        input_coords = inputs[0]
        
        prepped_coords = self.coordconvprep(input_coords)
        self.a('prepped_coords', prepped_coords)

        logits = self.model(prepped_coords)
        logits = tf.identity(logits, name='logits') # just to rename it

        logits_flat = Flatten()(logits)

        batch_size_tensor = tf.shape(input_coords)[0]  # get batch size
        
        prob_flat = tf.nn.softmax(logits_flat)
        prob = tf_reshape_like(prob_flat, logits, name='softmax_prob')
        
        pixelwise_prob_flat = tf.nn.sigmoid(logits_flat)
        pixelwise_prob = tf_reshape_like(pixelwise_prob_flat, logits, name='pixelwise_prob')

        assert len(inputs) == 2, "Not on-the-fly, supply a target image"
        labels = inputs[1]

        labels_flat = Flatten()(labels)

        self.a('prob', prob)
        self.a('prob_flat', prob_flat)
        self.a('pixelwise_prob', pixelwise_prob)
        self.a('pixelwise_prob_flat', pixelwise_prob_flat)
        self.a('logits', logits)
        self.a('logits_flat', logits_flat)
        self.a('labels', labels)
        self.a('labels_flat', labels_flat)

        self.make_losses_and_metrics()
        return logits

    def make_losses_and_metrics(self):

        xe_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits_flat, labels=self.labels_flat))
      
        mse_loss = tf.reduce_mean(
                    tf.pow(self.logits_flat - self.labels_flat, 2))

        sigm_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits_flat, labels=self.labels_flat))
        
        # regularizers
        reg_loss = tf.losses.get_regularization_loss()
        reg_losses = tf.losses.get_regularization_losses()

        argmax_prob = tf.argmax(self.prob_flat, 1)   # index in [0,64*64)
        # convert indices to 2D coordinates
        argmax_x = argmax_prob // self.x_dim
        argmax_y = argmax_prob % self.x_dim

        argmax_label = tf.argmax(self.labels_flat, 1)

        argmax_x_l = tf.floordiv(argmax_label, self.y_dim)
        argmax_y_l = tf.mod(argmax_label, self.y_dim)

        self.a('argmax_prob', argmax_prob)
        self.a('argmax_label', argmax_label)
        self.a('argmax_x', argmax_x)
        self.a('argmax_y', argmax_y)
        self.a('argmax_x_l', argmax_x_l)
        self.a('argmax_y_l', argmax_y_l)


        correct = tf.equal(argmax_prob, argmax_label)
        accuracy = tf.reduce_mean(tf.to_float(correct))
        eucl_dist = tf.reduce_mean(tf.sqrt(tf.to_float(tf.square(argmax_x-argmax_x_l) + tf.square(argmax_y-argmax_y_l))))
        manh_dist = tf.reduce_mean(tf.to_float(tf.abs(argmax_x-argmax_x_l) + tf.abs(argmax_y-argmax_y_l)))

        self.a('reg_losses', reg_losses)
        self.a('correct', correct)
        self.a('accuracy', accuracy, trackable=True)
        self.a('eucl_dist', eucl_dist, trackable=True)
        self.a('manh_dist', manh_dist, trackable=True)
        self.a('xe_loss', xe_loss, trackable=True)
        self.a('mse_loss', mse_loss, trackable=True)
        self.a('sigm_loss', sigm_loss, trackable=True)
        self.a('reg_loss', reg_loss, trackable=True)

        # calculate IOU, useful if input_label is an image

        painted = tf.round(self.pixelwise_prob_flat)  # 1 if > 0.5, 0 if <= 0.5
        n_painted = tf.reduce_mean(tf.reduce_sum(painted, 1))
        
        n_intersection = tf.reduce_sum(tf.multiply(painted, self.labels_flat), 1) # num of pixels in intersection
        
        n_union = tf.reduce_sum(tf.to_float(tf.logical_or(tf.cast(painted, tf.bool), tf.cast(self.labels_flat, tf.bool))), 1)

        self.a('painted', painted)
        self.a('n_intersection', n_intersection)
        self.a('n_union', n_union)

        intersect_pixs = tf.reduce_mean(n_intersection)
        iou = tf.reduce_mean(tf.div(n_intersection, n_union))

        self.a('painted_pixs', n_painted, trackable=True)
        self.a('intersect_pixs', intersect_pixs, trackable=True)
        self.a('iou', iou, trackable=True)


        if self.use_mse_loss:
            self.a('loss', mse_loss+reg_loss, trackable=True)
        elif self.use_sigm_loss:
            self.a('loss', sigm_loss+reg_loss, trackable=True)
        else:
            self.a('loss', xe_loss+reg_loss, trackable=True)

        return

class UpsampleConvPainter(Layers):
    '''A upsample+conv model that paints a pixel as directed by x,y coord inputs'''

    def __init__(self, l2=0, x_dim=64, y_dim=64, fs=3, mul=1, coordconv=False, 
                 include_r=False, use_mse_loss=False, use_sigm_loss=False):
        super(UpsampleConvPainter, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.use_mse_loss = use_mse_loss
        self.use_sigm_loss = use_sigm_loss
        
        with tf.variable_scope("model"):
            if coordconv:
                self.l('model', SequentialNetwork([
                       Lambda(lambda xx: tf.cast(xx, 'float32')),
                       Lambda(lambda xx: tf.reshape(xx, [-1, 1, 1, 2])),
                       UpSampling2D(size=(2,2)),
                           # output_shape=[None, 2, 2, 2]
                       CoordConv(2, 2, False, 64*mul, (fs,fs), padding='same',
                           kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)
                           ),
                           # output_shape=[None, 2, 2, 4]
                       BatchNormalization(momentum=0.9, epsilon=1e-5),
                       ReLu,
                           # output_shape=[None, 2, 2, 64]
                       UpSampling2D(size=(2,2)),  # output_shape=[None, 4, 4, 64]
                       AddCoords(x_dim=4, y_dim=4, with_r=include_r, skiptile=True),
                       Conv2D(64*mul, (fs,fs), padding='same',
                           kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                       BatchNormalization(momentum=0.9, epsilon=1e-5),
                       ReLu,
                           # output_shape=[None, 4, 4, 64]
                       UpSampling2D(size=(2,2)),  # output_shape=[None, 8, 8, 64]
                       AddCoords(x_dim=8, y_dim=8, with_r=include_r, skiptile=True),
                       Conv2D(64*mul, (fs,fs), padding='same',
                           kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                           # output_shape=[None, 8, 8, 64]
                       BatchNormalization(momentum=0.9, epsilon=1e-5),
                       ReLu,
                       UpSampling2D(size=(2,2)),  # output_shape=[None, 16, 16, 64]
                       AddCoords(x_dim=16, y_dim=16, with_r=include_r, skiptile=True),
                       Conv2D(32*mul, (fs,fs), padding='same',
                           kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                           # output_shape=[None, 16, 16, 32]
                       BatchNormalization(momentum=0.9, epsilon=1e-5),
                       ReLu,
                       UpSampling2D(size=(2,2)),  # output_shape=[None, 32, 32, 32]
                       AddCoords(x_dim=32, y_dim=32, with_r=include_r, skiptile=True),
                       Conv2D(32*mul, (fs,fs), padding='same',
                           kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                           # output_shape=[None, 32, 32, 32]
                       BatchNormalization(momentum=0.9, epsilon=1e-5),
                       ReLu,
                       UpSampling2D(size=(2,2)),  # output_shape=[None, 64, 64, 32]
                       AddCoords(x_dim=64, y_dim=64, with_r=include_r, skiptile=True),
                       Conv2D(1, (fs,fs), padding='same',
                           kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                           # output_shape=[None, 64, 64, 1]
                       ], name='upsample_coordconv_painter'))
            else:
                self.l('model', SequentialNetwork([
                       Lambda(lambda xx: tf.cast(xx, 'float32')),
                       Lambda(lambda xx: tf.reshape(xx, [-1, 1, 1, 2])),
                       UpSampling2D(size=(2,2)),
                           # output_shape=[None, 2, 2, 2]
                       Conv2D(64*mul, (fs,fs), padding='same',
                           kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                       BatchNormalization(momentum=0.9, epsilon=1e-5),
                       ReLu,
                           # output_shape=[None, 2, 2, 64]
                       UpSampling2D(size=(2,2)),  # output_shape=[None, 4, 4, 64]
                       Conv2D(64*mul, (fs,fs), padding='same',
                           kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                       BatchNormalization(momentum=0.9, epsilon=1e-5),
                       ReLu,
                           # output_shape=[None, 4, 4, 64]
                       UpSampling2D(size=(2,2)),  # output_shape=[None, 8, 8, 64]
                       Conv2D(64*mul, (fs,fs), padding='same',
                           kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                           # output_shape=[None, 8, 8, 64]
                       BatchNormalization(momentum=0.9, epsilon=1e-5),
                       ReLu,
                       UpSampling2D(size=(2,2)),  # output_shape=[None, 16, 16, 64]
                       Conv2D(32*mul, (fs,fs), padding='same',
                           kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                           # output_shape=[None, 16, 16, 32]
                       BatchNormalization(momentum=0.9, epsilon=1e-5),
                       ReLu,
                       UpSampling2D(size=(2,2)),  # output_shape=[None, 32, 32, 32]
                       Conv2D(32*mul, (fs,fs), padding='same',
                           kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                           # output_shape=[None, 32, 32, 32]
                       BatchNormalization(momentum=0.9, epsilon=1e-5),
                       ReLu,
                       UpSampling2D(size=(2,2)),  # output_shape=[None, 64, 64, 32]
                       Conv2D(1, (fs,fs), padding='same',
                           kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                           # output_shape=[None, 64, 64, 1]
                       ], name='upsample_conv_painter'))
        return 
    
    def call(self, inputs):
        # Shapes:
        # input_coords: (batch, 2)
        # logits: (batch, x_dim, y_dim, 1)
        # logits_flat: (batch, x_dim*y_dim)
        
        input_coords = inputs[0]

        logits = self.model([input_coords])
        logits = tf.identity(logits, name='logits') # just to rename it

        logits_flat = Flatten()(logits)

        batch_size_tensor = tf.shape(input_coords)[0]  # get batch size
        
        prob_flat = tf.nn.softmax(logits_flat)
        prob = tf_reshape_like(prob_flat, logits, name='softmax_prob')
        
        pixelwise_prob_flat = tf.nn.sigmoid(logits_flat)
        pixelwise_prob = tf_reshape_like(pixelwise_prob_flat, logits, name='pixelwise_prob')

        assert len(inputs) == 2, "Not on-the-fly, supply a target image"
        labels = inputs[1]

        labels_flat = Flatten()(labels)

        self.a('prob', prob)
        self.a('prob_flat', prob_flat)
        self.a('pixelwise_prob', pixelwise_prob)
        self.a('pixelwise_prob_flat', pixelwise_prob_flat)
        self.a('logits', logits)
        self.a('logits_flat', logits_flat)
        self.a('labels', labels)
        self.a('labels_flat', labels_flat)

        self.make_losses_and_metrics()
        return logits

    def make_losses_and_metrics(self):

        xe_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits_flat, labels=self.labels_flat))
      
        mse_loss = tf.reduce_mean(
                    tf.pow(self.logits_flat - self.labels_flat, 2))

        sigm_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits_flat, labels=self.labels_flat))
        
        # regularizers
        reg_loss = tf.losses.get_regularization_loss()
        reg_losses = tf.losses.get_regularization_losses()

        argmax_prob = tf.argmax(self.prob_flat, 1)   # index in [0,64*64)
        # convert indices to 2D coordinates
        argmax_x = argmax_prob // self.x_dim
        argmax_y = argmax_prob % self.x_dim

        argmax_label = tf.argmax(self.labels_flat, 1)

        argmax_x_l = tf.floordiv(argmax_label, self.y_dim)
        argmax_y_l = tf.mod(argmax_label, self.y_dim)

        self.a('argmax_prob', argmax_prob)
        self.a('argmax_label', argmax_label)
        self.a('argmax_x', argmax_x)
        self.a('argmax_y', argmax_y)
        self.a('argmax_x_l', argmax_x_l)
        self.a('argmax_y_l', argmax_y_l)


        correct = tf.equal(argmax_prob, argmax_label)
        accuracy = tf.reduce_mean(tf.to_float(correct))
        eucl_dist = tf.reduce_mean(tf.sqrt(tf.to_float(tf.square(argmax_x-argmax_x_l) + tf.square(argmax_y-argmax_y_l))))
        manh_dist = tf.reduce_mean(tf.to_float(tf.abs(argmax_x-argmax_x_l) + tf.abs(argmax_y-argmax_y_l)))

        self.a('reg_losses', reg_losses)
        self.a('correct', correct)
        self.a('accuracy', accuracy, trackable=True)
        self.a('eucl_dist', eucl_dist, trackable=True)
        self.a('manh_dist', manh_dist, trackable=True)
        self.a('xe_loss', xe_loss, trackable=True)
        self.a('mse_loss', mse_loss, trackable=True)
        self.a('sigm_loss', sigm_loss, trackable=True)
        self.a('reg_loss', reg_loss, trackable=True)

        # calculate IOU, useful if input_label is an image

        painted = tf.round(self.pixelwise_prob_flat)  # 1 if > 0.5, 0 if <= 0.5
        n_painted = tf.reduce_mean(tf.reduce_sum(painted, 1))
        
        n_intersection = tf.reduce_sum(tf.multiply(painted, self.labels_flat), 1) # num of pixels in intersection
        
        n_union = tf.reduce_sum(tf.to_float(tf.logical_or(tf.cast(painted, tf.bool), tf.cast(self.labels_flat, tf.bool))), 1)

        self.a('painted', painted)
        self.a('n_intersection', n_intersection)
        self.a('n_union', n_union)

        intersect_pixs = tf.reduce_mean(n_intersection)
        iou = tf.reduce_mean(tf.div(n_intersection, n_union))

        self.a('painted_pixs', n_painted, trackable=True)
        self.a('intersect_pixs', intersect_pixs, trackable=True)
        self.a('iou', iou, trackable=True)


        if self.use_mse_loss:
            self.a('loss', mse_loss+reg_loss, trackable=True)
        elif self.use_sigm_loss:
            self.a('loss', sigm_loss+reg_loss, trackable=True)
        else:
            self.a('loss', xe_loss+reg_loss, trackable=True)

        return



class CoordConvImagePainter(Layers):
    '''A CoordConv that paints an image  as directed by x,y coord inputs'''

    def __init__(self, l2=0, fs=3, x_dim=64, y_dim=64, mul=1, include_r=False, 
            use_mse_loss=False, use_sigm_loss=True, interm_loss=None,
            no_softmax=False, version='working'):
        super(CoordConvImagePainter, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.use_mse_loss = use_mse_loss
        self.use_sigm_loss = use_sigm_loss
        self.interm_loss = interm_loss
        
        self.l('coordconvprep', SequentialNetwork([
            Lambda(lambda xx: tf.reshape(xx, [-1, 1, 1, 2])),
            AddCoords(x_dim=x_dim, y_dim=y_dim, with_r=include_r, skiptile=False), # (batch, 64, 64, 4 or 5)
            ], name='coordconvprep'))
            
        self.l('coordconvmodel', SequentialNetwork([
            Conv2D(32, (1,1), padding='valid',
                kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
            ReLu,
            Conv2D(32, (1,1), padding='valid',
                kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
            ReLu,
            Conv2D(64, (1,1), padding='valid',
                kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
            ReLu,
            Conv2D(64, (1,1), padding='valid',
                kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
            ReLu,
            Conv2D(1, (1,1), padding='same',
                kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
            ], name='coordconvmodel'))

        self.l('sharpen', SequentialNetwork([
            Flatten(), # (batch, -1)
            Softmax,
            Lambda(lambda xx: tf.reshape(xx, [-1, x_dim, y_dim, 1])),
            ], name='sharpen'))

        if version == 'simple':
            net = build_simple_one_channel_onehot2image(l2, name='convmodel')
            self.l('convmodel', net)

        elif version == 'working':
            net = build_working_conv_onehot2image(l2, mul, fs, name='convmodel')
            self.l('convmodel', net)
        
        if no_softmax: 
            self.l('model', SequentialNetwork([
                ('coordconvprep', self.coordconvprep),
                ('coordconvmodel', self.coordconvmodel),
                ('convmodel', self.convmodel)
                ], name='model'))
        else:
            self.l('model', SequentialNetwork([
                ('coordconvprep', self.coordconvprep),
                ('coordconvmodel', self.coordconvmodel),
                ('sharpen', self.sharpen),
                ('convmodel', self.convmodel)
                ], name='model'))

        return 
    
    def call(self, inputs):
        # Shapes:
        # input_coords: (batch, 2)
        # logits: (batch, x_dim, y_dim, 1)
        # logits_flat: (batch, x_dim*y_dim)
        
        if len(inputs) == 2:
            input_coords, input_images = inputs[0], inputs[1]
        elif len(inputs) == 3:
            input_coords, input_1hot, input_images = inputs[0], inputs[1], inputs[2]
        else:
            raise ValueError('model requires either 2 or 3 tensors: input_coords, (input_1hot,) input_images')
        
        prepped_coords = self.coordconvprep(input_coords)
        center_logits = self.coordconvmodel(prepped_coords)
        center_logits = tf.identity(center_logits, name='center_logits') # just to rename it
        center_logits_flat = Flatten()(center_logits)
        sharpened_logits = self.sharpen(center_logits)
        sharpened_logits = tf.identity(sharpened_logits, name='sharpened_logits') # just to rename it

        self.a('prepped_coords', prepped_coords)
        self.a('center_logits', center_logits)
        self.a('center_logits_flat', center_logits_flat)
        self.a('sharpened_logits', sharpened_logits)

        logits = self.model(input_coords)
        #logits = self.convmodel(input_1hot) # HACK to see if second part of the model works
        logits = tf.identity(logits, name='logits') # just to rename it

        logits_flat = Flatten()(logits)

        center_prob = tf.identity(sharpened_logits, name='prob') # just to rename it
        center_prob_flat = Flatten()(center_prob)
        
        pixelwise_prob_flat = tf.nn.sigmoid(logits_flat)
        pixelwise_prob = tf_reshape_like(pixelwise_prob_flat, logits, name='pixelwise_prob')

        images_flat = Flatten()(input_images)

        self.a('center_prob', center_prob)
        self.a('center_prob_flat', center_prob_flat)
        self.a('pixelwise_prob', pixelwise_prob)
        self.a('pixelwise_prob_flat', pixelwise_prob_flat)
        self.a('logits', logits)
        self.a('logits_flat', logits_flat)
        self.a('images_flat', images_flat)
        
        if 'input_1hot' in locals():
            onehot_flat = Flatten()(input_1hot)
            self.a('onehot_flat', onehot_flat)

        self.make_losses_and_metrics()
        return logits

    def make_losses_and_metrics(self):

        # intermediate loss
        if hasattr(self, 'onehot_flat'):
            interm_softmax_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.center_logits_flat, labels=self.onehot_flat))
            interm_mse_loss = tf.reduce_mean(
                    tf.pow(self.center_logits_flat - self.onehot_flat, 2))
            self.a('interm_softmax_loss', interm_softmax_loss, trackable=True)
            self.a('interm_mse_loss', interm_mse_loss, trackable=True)
      
        # losses that have to do with only final images
        mse_loss = tf.reduce_mean(
                    tf.pow(self.logits_flat - self.images_flat, 2))

        sigm_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits_flat, labels=self.images_flat))
        
        # regularizers
        reg_loss = tf.losses.get_regularization_loss()
        reg_losses = tf.losses.get_regularization_losses()

        argmax_prob = tf.argmax(self.center_prob_flat, 1)   # index in [0,64*64)
        ## convert indices to 2D coordinates
        argmax_x = argmax_prob // self.x_dim
        argmax_y = argmax_prob % self.x_dim
        self.a('argmax_prob', argmax_prob)
        self.a('argmax_x', argmax_x)
        self.a('argmax_y', argmax_y)

        if hasattr(self, 'onehot_flat'):
            argmax_label = tf.argmax(self.onehot_flat, 1)
            argmax_x_l = tf.floordiv(argmax_label, self.y_dim)
            argmax_y_l = tf.mod(argmax_label, self.y_dim)
            self.a('argmax_label', argmax_label)
            self.a('argmax_x_l', argmax_x_l)
            self.a('argmax_y_l', argmax_y_l)

            correct = tf.equal(argmax_prob, argmax_label)
            self.a('correct', correct)

            accuracy = tf.reduce_mean(tf.to_float(correct))
            eucl_dist = tf.reduce_mean(tf.sqrt(tf.to_float(tf.square(argmax_x-argmax_x_l) + tf.square(argmax_y-argmax_y_l))))
            manh_dist = tf.reduce_mean(tf.to_float(tf.abs(argmax_x-argmax_x_l) + tf.abs(argmax_y-argmax_y_l)))
            self.a('accuracy', accuracy, trackable=True)
            self.a('eucl_dist', eucl_dist, trackable=True)
            self.a('manh_dist', manh_dist, trackable=True)

        self.a('reg_losses', reg_losses)
        self.a('mse_loss', mse_loss, trackable=True)
        self.a('sigm_loss', sigm_loss, trackable=True)
        self.a('reg_loss', reg_loss, trackable=True)

        # calculate IOU, useful if input_label is an image

        painted = tf.round(self.pixelwise_prob_flat)  # 1 if > 0.5, 0 if <= 0.5
        n_painted = tf.reduce_mean(tf.reduce_sum(painted, 1))

        n_intersection = tf.reduce_sum(tf.multiply(painted, self.images_flat), 1) # num of pixels in intersection
        n_union = tf.reduce_sum(
                tf.to_float(tf.logical_or(tf.cast(painted, tf.bool), 
                            tf.cast(self.images_flat, tf.bool))),
                1)

        self.a('painted', painted)
        self.a('n_intersection', n_intersection)
        self.a('n_union', n_union)

        intersect_pixs = tf.reduce_mean(n_intersection)
        iou = tf.reduce_mean(tf.div(n_intersection, n_union))

        self.a('painted_pixs', n_painted, trackable=True)
        self.a('intersect_pixs', intersect_pixs, trackable=True)
        self.a('iou', iou, trackable=True)


        if self.use_mse_loss:
            loss = mse_loss+reg_loss
        elif self.use_sigm_loss:
            loss = sigm_loss+reg_loss
        else:
            raise ValueError('use either sigmoid or mse loss')
       
        if self.interm_loss is not None:
            if self.interm_loss == 'mse':
                loss += interm_mse_loss
            elif self.interm_loss == 'softmax':
                loss += interm_softmax_loss
            else:
                raise ValueError('Support only `mse` or `softmax` intermediate loss')

        self.a('loss', loss, trackable=True)

        return

class DeconvBottleneckPainter(Layers):
    '''
    Like DeconvPainter but squeeze to 1 channel in between.
    Option to either sharpen that channel or not,
    with enforced loss or with softmax layer
    '''

    def __init__(self, l2=0, x_dim=64, y_dim=64, fs=3, mul=1, 
                 use_mse_loss=False, use_sigm_loss=False,
                 interm_loss=None, no_softmax=False,
                 version='working'):
        super(DeconvBottleneckPainter, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.use_mse_loss = use_mse_loss
        self.use_sigm_loss = use_sigm_loss
        self.interm_loss = interm_loss
       

        net = build_deconv_coords2image(l2, mul, fs, name='coords2center')
        self.l('coords2center', net)

        self.l('sharpen', SequentialNetwork([
                Flatten(), # (batch, -1)
                Softmax,
                Lambda(lambda xx: tf.reshape(xx, [-1, x_dim, y_dim, 1])),
                ], name="sharpen"))
            
        if version == 'simple':
            net = build_simple_one_channel_onehot2image(l2, name='center2image')
            self.l('center2image', net)

        elif version == 'working':
            net = build_working_conv_onehot2image(l2, mul, fs, name='center2image')
            self.l('center2image', net)

            if no_softmax:
                self.l('model', SequentialNetwork([
                        ('coords2center', self.coords2center),
                        ('center2image', self.center2image)
                        ], name='model'))
            else:
                self.l('model', SequentialNetwork([
                        ('coords2center', self.coords2center),
                        ('sharpen', self.sharpen),
                        ('center2image', self.center2image)
                        ], name='model'))

        return 
    
    def call(self, inputs):
       
        if len(inputs) == 2:
            input_coords, input_images = inputs[0], inputs[1]
        elif len(inputs) == 3:
            input_coords, input_1hot, input_images = inputs[0], inputs[1], inputs[2]
        else:
            raise ValueError('model requires either 2 or 3 tensors: input_coords, (input_1hot,) input_images')
        
        center_logits = self.coords2center(input_coords)
        center_logits = tf.identity(center_logits, name='center_logits') # just to rename it
        center_logits_flat = Flatten()(center_logits)
        sharpened_logits = self.sharpen(center_logits)
        sharpened_logits = tf.identity(sharpened_logits, name='sharpened_logits') # just to rename it
        
        self.a('center_logits', center_logits)
        self.a('center_logits_flat', center_logits_flat)
        self.a('sharpened_logits', sharpened_logits)
        

        logits = self.model(input_coords)
        logits = tf.identity(logits, name='logits') # just to rename it

        logits_flat = Flatten()(logits)

        center_prob = tf.identity(sharpened_logits, name='prob') # just to rename it
        center_prob_flat = Flatten()(center_prob)
        
        pixelwise_prob_flat = tf.nn.sigmoid(logits_flat)
        pixelwise_prob = tf_reshape_like(pixelwise_prob_flat, logits, name='pixelwise_prob')

        images_flat = Flatten()(input_images)

        self.a('center_prob', center_prob)
        self.a('center_prob_flat', center_prob_flat)
        self.a('pixelwise_prob', pixelwise_prob)
        self.a('pixelwise_prob_flat', pixelwise_prob_flat)
        self.a('logits', logits)
        self.a('logits_flat', logits_flat)
        self.a('images_flat', images_flat)
        
        if 'input_1hot' in locals():
            onehot_flat = Flatten()(input_1hot)
            self.a('onehot_flat', onehot_flat)

        self.make_losses_and_metrics()

        return logits

    def make_losses_and_metrics(self):
        # intermediate loss
        if hasattr(self, 'onehot_flat'):
            interm_softmax_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.center_logits_flat, labels=self.onehot_flat))
            interm_mse_loss = tf.reduce_mean(
                    tf.pow(self.center_logits_flat - self.onehot_flat, 2))
            self.a('interm_softmax_loss', interm_softmax_loss, trackable=True)
            self.a('interm_mse_loss', interm_mse_loss, trackable=True)
      
        # losses that have to do with only final images
        mse_loss = tf.reduce_mean(
                    tf.pow(self.logits_flat - self.images_flat, 2))

        sigm_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits_flat, labels=self.images_flat))
        
        # regularizers
        reg_loss = tf.losses.get_regularization_loss()
        reg_losses = tf.losses.get_regularization_losses()

        argmax_prob = tf.argmax(self.center_prob_flat, 1)   # index in [0,64*64)
        ## convert indices to 2D coordinates
        argmax_x = argmax_prob // self.x_dim
        argmax_y = argmax_prob % self.x_dim
        self.a('argmax_prob', argmax_prob)
        self.a('argmax_x', argmax_x)
        self.a('argmax_y', argmax_y)

        if hasattr(self, 'onehot_flat'):
            argmax_label = tf.argmax(self.onehot_flat, 1)
            argmax_x_l = tf.floordiv(argmax_label, self.y_dim)
            argmax_y_l = tf.mod(argmax_label, self.y_dim)
            self.a('argmax_label', argmax_label)
            self.a('argmax_x_l', argmax_x_l)
            self.a('argmax_y_l', argmax_y_l)

            correct = tf.equal(argmax_prob, argmax_label)
            self.a('correct', correct)

            accuracy = tf.reduce_mean(tf.to_float(correct))
            eucl_dist = tf.reduce_mean(tf.sqrt(tf.to_float(tf.square(argmax_x-argmax_x_l) + tf.square(argmax_y-argmax_y_l))))
            manh_dist = tf.reduce_mean(tf.to_float(tf.abs(argmax_x-argmax_x_l) + tf.abs(argmax_y-argmax_y_l)))
            self.a('accuracy', accuracy, trackable=True)
            self.a('eucl_dist', eucl_dist, trackable=True)
            self.a('manh_dist', manh_dist, trackable=True)

        self.a('reg_losses', reg_losses)
        self.a('mse_loss', mse_loss, trackable=True)
        self.a('sigm_loss', sigm_loss, trackable=True)
        self.a('reg_loss', reg_loss, trackable=True)

        # calculate IOU, useful if input_label is an image

        painted = tf.round(self.pixelwise_prob_flat)  # 1 if > 0.5, 0 if <= 0.5
        n_painted = tf.reduce_mean(tf.reduce_sum(painted, 1))

        n_intersection = tf.reduce_sum(tf.multiply(painted, self.images_flat), 1) # num of pixels in intersection
        n_union = tf.reduce_sum(
                tf.to_float(tf.logical_or(tf.cast(painted, tf.bool), 
                            tf.cast(self.images_flat, tf.bool))),
                1)

        self.a('painted', painted)
        self.a('n_intersection', n_intersection)
        self.a('n_union', n_union)

        intersect_pixs = tf.reduce_mean(n_intersection)
        iou = tf.reduce_mean(tf.div(n_intersection, n_union))

        self.a('painted_pixs', n_painted, trackable=True)
        self.a('intersect_pixs', intersect_pixs, trackable=True)
        self.a('iou', iou, trackable=True)


        if self.use_mse_loss:
            loss = mse_loss+reg_loss
        elif self.use_sigm_loss:
            loss = sigm_loss+reg_loss
        else:
            raise ValueError('use either sigmoid or mse loss')
       
        if self.interm_loss is not None:
            if self.interm_loss == 'mse':
                loss += interm_mse_loss
            elif self.interm_loss == 'softmax':
                loss += interm_softmax_loss
            else:
                raise ValueError('Support only `mse` or `softmax` intermediate loss')

        self.a('loss', loss, trackable=True)
        return



def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def build_working_conv_onehot2image(l2, mul, fs, name=''):
    net = SequentialNetwork([
                Conv2D(8*mul, (fs,fs), padding='same',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                #BatchNormalization(momentum=0.9, epsilon=1e-5),
                ReLu,
                Conv2D(8*mul, (fs,fs), padding='same',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                #BatchNormalization(momentum=0.9, epsilon=1e-5),
                ReLu,
                Conv2D(16*mul, (fs,fs), padding='same',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                #BatchNormalization(momentum=0.9, epsilon=1e-5),
                ReLu,
                Conv2D(16*mul, (fs,fs), padding='same',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                #BatchNormalization(momentum=0.9, epsilon=1e-5),
                ReLu,
                Conv2D(1, (fs,fs), padding='same',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                ], name=name)
    return net


class CoordConvGAN(Layers):
    '''basically DCGAN64 structure, enhanced with coordconv-like coordinate inputs'''

    def __init__(self, l2=0, x_dim=64, y_dim=64, cout=1,
                 add_r=False, coords_in_g=False, coords_in_d=False):
        
        super(CoordConvGAN, self).__init__()
        #self.l2 = l2
        #self.x_dim = x_dim
        #self.y_dim = y_dim
        #self.add_r = add_r
        self.coords_in_g = coords_in_g
        self.coords_in_d = coords_in_d

        with tf.variable_scope("generator"):
            self.l('coordconvprep_g', SequentialNetwork([
                Lambda(lambda xx: tf.expand_dims(tf.expand_dims(xx,1),1)),  # (batch, z_dim) -> (batch, 1, 1, z_dim)
                AddCoords(x_dim=x_dim, y_dim=y_dim, with_r=add_r, skiptile=False), # (batch, 64, 64, 4 or 5)
                ], name='coordconvprep_g'))

            self.l('coordconv_generator', SequentialNetwork([
                Conv2D(8*64, (1,1), padding='valid',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                ReLu,
                Conv2D(64*4, (1,1), padding='valid',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                ReLu,
                Conv2D(64*4, (1,1), padding='valid',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                ReLu,
                Conv2D(64*2, (1,1), padding='same',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                ReLu,
                Conv2D(64*1, (1,1), padding='same',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                ReLu,
                Conv2D(64*1, (3,3), padding='same',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                ReLu,
                Conv2D(64*1, (3,3), padding='same',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                ReLu,
                Conv2D(cout, (1,1), padding='same',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                ], name='coordconv_generator'))
        
            self.l('deconv_generator', SequentialNetwork([
                Dense(4*4*8*64),
                Lambda(lambda xx: tf.reshape(xx, [-1, 4, 4, 8*64])),
                BatchNormalization(momentum=0.9, epsilon=1e-5),
                ReLu,
                Deconv(64*4, (5,5), (2,2), padding='same'),          # output_shape=[None, 8, 8, 512]
                BatchNormalization(momentum=0.9, epsilon=1e-5),
                ReLu,
                Deconv(64*2, (5,5), (2,2), padding='same'),          # output_shape=[None, 16, 16, 256]
                BatchNormalization(momentum=0.9, epsilon=1e-5),
                ReLu,
                Deconv(64*1, (5,5), (2,2), padding='same'),          # output_shape=[None, 32, 32, 128]
                BatchNormalization(momentum=0.9, epsilon=1e-5),
                ReLu,
                Deconv(cout, (5,5), (2,2), padding='same'),            # output_shape=[None, 64, 64, 3]
                Tanh
                ], name='deconv_generator'))

        with tf.variable_scope("discriminator"):
            
            self.l('coordconvprep_d', SequentialNetwork([
                AddCoords(x_dim=x_dim, y_dim=y_dim, with_r=add_r, skiptile=True), # (batch, 64, 64, 4 or 5)
                ], name='coordconvprep_d'))

            self.l('discriminator_minus_last', SequentialNetwork([
                Conv2D(64, (5,5), 2, padding='same', activation=lrelu),
                Conv2D(128, (5,5), 2, padding='same'),
                BatchNormalization(momentum=0.9, epsilon=1e-5),
                LReLu,
                Conv2D(256, (5,5), 2, padding='same'),
                BatchNormalization(momentum=0.9, epsilon=1e-5),
                LReLu,
                Conv2D(512, (5,5), 2, padding='same'),
                BatchNormalization(momentum=0.9, epsilon=1e-5),
                LReLu,
                Flatten(),
                #Dense(1)
                ], name='discriminator_minus_last'))
            

            self.l('discriminator', SequentialNetwork(
                [self.discriminator_minus_last,
                Dense(1)],
                name='discriminator'))

    def call(self, inputs, feature_matching_loss=False, feature_match_loss_weight=1.0):
        
        assert len(inputs) == 2+(feature_matching_loss), \
                'inputs: images, noises, (images2 if feature matching)'
        
        input_x, input_z = inputs[0], inputs[1]
        
        if feature_matching_loss:
            input_x2 = inputs[2]

        #self.build_model()

        # call generator
        if self.coords_in_g:
            enhanced_z = self.coordconvprep_g(input_z)
            g_out = self.coordconv_generator(enhanced_z)
            self.a('enhanced_z', enhanced_z)
        else:
            g_out = self.deconv_generator(input_z)
       
        self.a('g_out', g_out)
        
        if self.coords_in_g:
            self.a('fake_images', Tanh(g_out))
        else:
            self.a('fake_images', g_out)

        
        # call discriminator
        if self.coords_in_d:
            enhanced_img = self.coordconvprep_d(input_x)
            enhanced_img_fake = self.coordconvprep_d(g_out)
            d_real_logits = self.discriminator(enhanced_img)
            d_fake_logits = self.discriminator(enhanced_img_fake)
            
            if feature_matching_loss:
                enhanced_img2 = self.coordconvprep_d(input_x2)
                d_real_features = self.discriminator_minus_last(enhanced_img2)
                d_fake_features = self.discriminator_minus_last(enhanced_img_fake)
        else:
            d_real_logits = self.discriminator(input_x)
            d_fake_logits = self.discriminator(g_out)
            if feature_matching_loss:
                d_real_features = self.discriminator_minus_last(input_x2)
                d_fake_features = self.discriminator_minus_last(g_out)

        self.a('d_fake_logits', d_fake_logits)
        self.a('d_real_logits', d_real_logits)

        # get loss for discriminator
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logits, labels=tf.ones_like(d_real_logits)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.zeros_like(d_fake_logits)))

        self.a('d_loss_real', d_loss_real, trackable=True)
        self.a('d_loss_fake', d_loss_fake, trackable=True)

        # get loss for generator
        g_loss_basic = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.ones_like(d_fake_logits)))
        self.a('g_loss_basic', g_loss_basic, trackable=True)
       
        # regularization loss
        reg_loss = tf.losses.get_regularization_loss()
        reg_losses = tf.losses.get_regularization_losses()
        self.a('reg_losses', reg_losses)
        self.a('reg_loss', reg_loss, trackable=True)


        if feature_matching_loss:
            g_loss_feture_match = tf.reduce_mean(tf.abs(tf.subtract(d_real_features, d_fake_features)))
            self.a('g_loss_feture_match', g_loss_feture_match, trackable=True)
            
            self.a('g_loss', g_loss_feture_match * feature_match_loss_weight + g_loss_basic + reg_loss, trackable=True)
            self.a('d_loss', d_loss_real + d_loss_fake, trackable=True)

        else:
            self.a('g_loss', g_loss_basic + reg_loss, trackable=True)
            self.a('d_loss', d_loss_real + d_loss_fake, trackable=True)
        
        # correct rate for discriminator
        pred_correct_real = tf.greater(d_real_logits, tf.zeros_like(d_real_logits))
        correct_real = tf.reduce_mean(tf.to_float(pred_correct_real))
                
        pred_correct_fake = tf.less(d_fake_logits, tf.zeros_like(d_fake_logits))
        correct_fake = tf.reduce_mean(tf.to_float(pred_correct_fake))

        self.a('correct_real', correct_real, trackable=True)
        self.a('correct_fake', correct_fake, trackable=True)

        return g_out








def build_simple_one_channel_onehot2image(l2, name=''):
    net = SequentialNetwork([
                Conv2D(1, (9,9), padding='same',
                    #kernel_initializer=he_normal, 
                    kernel_initializer=tf.ones_initializer(), 
                    bias_initializer=tf.constant_initializer([0.]),
                    kernel_regularizer=l2reg(l2)),  
                ], name=name)
    return net

def build_deconv_coords2image(l2, mul, fs, name=''):
    net = SequentialNetwork([
                Lambda(lambda xx: tf.cast(xx, 'float32')),
                Lambda(lambda xx: tf.reshape(xx, [-1, 1, 1, 2])),
                Deconv(64*mul, (fs,fs), (2,2), padding='same',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                     # output_shape=[None, 2, 2, 64]
                BatchNormalization(momentum=0.9, epsilon=1e-5),
                ReLu,
                Deconv(64*mul, (fs,fs), (2,2), padding='same',  
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                    # output_shape=[None, 4, 4, 64]
                BatchNormalization(momentum=0.9, epsilon=1e-5),
                ReLu,
                Deconv(64*mul, (fs,fs), (2,2), padding='same',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                    # output_shape=[None, 8, 8, 64]
                BatchNormalization(momentum=0.9, epsilon=1e-5),
                ReLu,
                Deconv(32*mul, (fs,fs), (2,2), padding='same',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                    # output_shape=[None, 16, 16, 32]
                BatchNormalization(momentum=0.9, epsilon=1e-5),
                ReLu,
                Deconv(32*mul, (fs,fs), (2,2), padding='same',  
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),  
                    # output_shape=[None, 32, 32, 32]
                BatchNormalization(momentum=0.9, epsilon=1e-5),
                ReLu,
                Deconv(1, (fs,fs), (2,2), padding='same',  
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                    # output_shape=[None, 64, 64, 1]
                ], name=name)
    return net

