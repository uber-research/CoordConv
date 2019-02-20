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

import sys
import os
import gzip
import cPickle as pickle
import numpy as np
import h5py
import itertools
import random
from IPython import embed
import colorama
import tensorflow as tf

from general.util import tic, toc, tic2, toc2, tic3, toc3, mkdir_p, WithTimer
from general.tfutil import (get_collection_intersection_summary, log_scalars,
                            sess_run_dict, summarize_weights, summarize_opt,
                            tf_assert_all_init, tf_get_uninitialized_variables,
                            add_grad_summaries, add_grads_and_vars_hist_summaries,
                            image_summaries_traintest)
from general.stats_buddy import StatsBuddy
from general.lr_policy import LRPolicyConstant, LRPolicyStep, LRPolicyValStep
from tf_plus import setup_session_and_seeds, learning_phase, print_trainable_warnings
from model_builders import (DeconvPainter, ConvImagePainter, ConvRegressor,
                            CoordConvPainter, CoordConvImagePainter,
                            DeconvBottleneckPainter, UpsampleConvPainter)
from util import *

# all choices of tasks/architectures, to be used for --arch

# Tasks reported in paper
# deconv_classification / coordconv_classification
#   input: coords
#   output:onehot image

# deconv_rendering / coordconv_rendering
#   input:  coords
#   output: full square image

# conv_regressor / coordconv_regressor
#   input: onehot image
#   output: coords

arch_choices = [  # tasks reported in  paper
    'deconv_classification',
    'deconv_rendering',
    'coordconv_classification',
    'coordconv_rendering',
    'conv_regressor',
    'coordconv_regressor',  # TODO add
    # addtiontal tasks
    'conv_onehot_image',
    'deconv_bottleneck',
    'upsample_conv_coords',
    'upsample_coordconv_coords']

lr_policy_choices = ('constant', 'step', 'valstep')
intermediate_loss_choices = (None, 'softmax', 'mse')


def main():
    parser = make_standard_parser(
        'Coordconv',
        arch_choices=arch_choices,
        skip_train=True,
        skip_val=True)
    # re-add train and val h5s as optional
    parser.add_argument('--data_h5', type=str,
                        default='../data/rectangle_4_uniform.h5',
                        help='data file in hdf5.')
    parser.add_argument('--x_dim', type=int, default=64,
                        help='x dimension of the output image')
    parser.add_argument('--y_dim', type=int, default=64,
                        help='y dimension of the output image')
    parser.add_argument('--lrpolicy', type=str, default='constant',
                        choices=lr_policy_choices, help='LR policy.')
    parser.add_argument('--lrstepratio', type=float,
                        default=.1, help='LR policy step ratio.')
    parser.add_argument('--lrmaxsteps', type=int, default=5,
                        help='LR policy step ratio.')
    parser.add_argument('--lrstepevery', type=int, default=50,
                        help='LR policy step ratio.')
    parser.add_argument('--filter_size', '-fs', type=int, default=3,
                        help='filter size in deconv network')
    parser.add_argument('--channel_mul', '-mul', type=int, default=2,
        help='Deconv model channel multiplier to make bigger models')
    parser.add_argument('--use_mse_loss', '-mse', action='store_true',
                        help='use mse loss instead of cross entropy')
    parser.add_argument('--use_sigm_loss', '-sig', action='store_true',
                        help='use sigmoid loss instead of cross entropy')
    parser.add_argument('--interm_loss', '-interm', default=None,
        choices=(None, 'softmax', 'mse'),
        help='add intermediate loss to end-to-end painter model')
    parser.add_argument('--no_softmax', '-nosfmx', action='store_true',
                        help='Remove softmax sharpening layer in model')

    args = parser.parse_args()

    if args.lrpolicy == 'step':
        lr_policy = LRPolicyStep(args)
    elif args.lrpolicy == 'valstep':
        lr_policy = LRPolicyValStep(args)
    else:
        lr_policy = LRPolicyConstant(args)

    minibatch_size = args.minibatch
    train_style, val_style = (
        '', '') if args.nocolor else (
        colorama.Fore.BLUE, colorama.Fore.MAGENTA)

    sess = setup_session_and_seeds(args.seed, assert_gpu=not args.cpu)

    # 0. Load data or generate data on the fly
    print 'Loading data: {}'.format(args.data_h5)

    if args.arch in ['deconv_classification',
                     'coordconv_classification',
                     'upsample_conv_coords',
                     'upsample_coordconv_coords']:

        # option a: generate data on the fly
        #data = list(itertools.product(range(args.x_dim),range(args.y_dim)))
        # random.shuffle(data)

        #train_test_split = .8
        #val_reps = int(args.x_dim * args.x_dim * train_test_split) // minibatch_size
        #val_size = val_reps * minibatch_size
        #train_end = args.x_dim * args.x_dim - val_size
        #train_x, val_x = np.array(data[:train_end]).astype('int'), np.array(data[train_end:]).astype('int')
        #train_y, val_y = None, None
        #DATA_GEN_ON_THE_FLY = True

        # option b: load the data
        fd = h5py.File(args.data_h5, 'r')

        train_x = np.array(fd['train_locations'], dtype=int)  # shape (2368, 2)
        train_y = np.array(fd['train_onehots'], dtype=float)  # shape (2368, 64, 64, 1)
        val_x = np.array(fd['val_locations'], dtype=float)  # shape (768, 2)
        val_y = np.array(fd['val_onehots'], dtype=float)  # shape (768, 64, 64, 1)
        DATA_GEN_ON_THE_FLY = False

        # number of image channels
        image_c = train_y.shape[-1] if train_y is not None and len(train_y.shape) == 4 else 1

    elif args.arch == 'conv_onehot_image':
        fd = h5py.File(args.data_h5, 'r')
        train_x = np.array(
            fd['train_onehots'],
            dtype=int)  # shape (2368, 64, 64, 1)
        train_y = np.array(fd['train_imagegray'],
                           dtype=float) / 255.0  # shape (2368, 64, 64, 1)
        val_x = np.array(
            fd['val_onehots'],
            dtype=float)  # shape (768, 64, 64, 1)
        val_y = np.array(fd['val_imagegray'], dtype=float) / \
            255.0  # shape (768, 64, 64, 1)

        image_c = train_y.shape[-1]

    elif args.arch == 'deconv_rendering':
        fd = h5py.File(args.data_h5, 'r')
        train_x = np.array(fd['train_locations'], dtype=int)  # shape (2368, 2)
        train_y = np.array(fd['train_imagegray'],
                           dtype=float) / 255.0  # shape (2368, 64, 64, 1)
        val_x = np.array(fd['val_locations'], dtype=float)  # shape (768, 2)
        val_y = np.array(fd['val_imagegray'], dtype=float) / \
            255.0  # shape (768, 64, 64, 1)

        image_c = train_y.shape[-1]

    elif args.arch == 'conv_regressor' or args.arch == 'coordconv_regressor':
        fd = h5py.File(args.data_h5, 'r')
        train_y = np.array(
            fd['train_normalized_locations'],
            dtype=float)  # shape (2368, 2)
        # /255.0 # shape (2368, 64, 64, 1)
        train_x = np.array(fd['train_onehots'], dtype=float)
        val_y = np.array(
            fd['val_normalized_locations'],
            dtype=float)  # shape (768, 2)
        val_x = np.array(
            fd['val_onehots'],
            dtype=float)  # shape (768, 64, 64, 1)

        image_c = train_x.shape[-1]

    elif args.arch == 'coordconv_rendering' or args.arch == 'deconv_bottleneck':
        fd = h5py.File(args.data_h5, 'r')
        train_x = np.array(fd['train_locations'], dtype=int)  # shape (2368, 2)
        train_y = np.array(fd['train_imagegray'],
                           dtype=float) / 255.0  # shape (2368, 64, 64, 1)
        val_x = np.array(fd['val_locations'], dtype=float)  # shape (768, 2)
        val_y = np.array(fd['val_imagegray'], dtype=float) / 255.0  # shape (768, 64, 64, 1)

        # add one-hot anyways to track accuracy etc. even if not used in loss
        train_onehot = np.array(
            fd['train_onehots'],
            dtype=int)  # shape (2368, 64, 64, 1)
        val_onehot = np.array(
            fd['val_onehots'],
            dtype=int)  # shape (768, 64, 64, 1)

        image_c = train_y.shape[-1]

    train_size = train_x.shape[0]
    val_size = val_x.shape[0]

    # 1. CREATE MODEL
    input_coords = tf.placeholder(
        shape=(None,2),
        dtype='float32',
        name='input_coords')  # cast later in model into float
    input_onehot = tf.placeholder(
        shape=(None, args.x_dim, args.y_dim, 1),
        dtype='float32',
        name='input_onehot')
    input_images = tf.placeholder(
        shape=(None, args.x_dim, args.y_dim, image_c),
        dtype='float32',
        name='input_images')

    if args.arch == 'deconv_classification':
        model = DeconvPainter(l2=args.l2, x_dim=args.x_dim, y_dim=args.y_dim,
                              fs=args.filter_size, mul=args.channel_mul,
                              onthefly=DATA_GEN_ON_THE_FLY,
                              use_mse_loss=args.use_mse_loss,
                              use_sigm_loss=args.use_sigm_loss)

        model.a('input_coords', input_coords)

        if not DATA_GEN_ON_THE_FLY:
            model.a('input_onehot', input_onehot)

        model([input_coords]) if DATA_GEN_ON_THE_FLY else model([input_coords, input_onehot])

    if args.arch == 'conv_regressor':
        regress_type = 'conv_uniform' if 'uniform' in args.data_h5 else 'conv_quarant'
        model = ConvRegressor(l2=args.l2, fs=args.filter_size, mul=args.channel_mul,
                              _type=regress_type)
        model.a('input_coords', input_coords)
        model.a('input_onehot', input_onehot)
        # call model on inputs
        model([input_onehot, input_coords])

    if args.arch == 'coordconv_regressor':
        model = ConvRegressor(l2=args.l2, fs=args.filter_size, mul=args.channel_mul,
                              _type='coordconv')
        model.a('input_coords', input_coords)
        model.a('input_onehot', input_onehot)
        # call model on inputs
        model([input_onehot, input_coords])

    if args.arch == 'conv_onehot_image':
        model = ConvImagePainter(l2=args.l2, fs=args.filter_size, mul=args.channel_mul,
            use_mse_loss=args.use_mse_loss, use_sigm_loss=args.use_sigm_loss,
            version='working')
            # version='simple') # version='simple' to hack a 9x9 all-ones filter solution
        model.a('input_onehot', input_onehot)
        model.a('input_images', input_images)
        # call model on inputs
        model([input_onehot, input_images])

    if args.arch == 'deconv_rendering':
        model = DeconvPainter(l2=args.l2, x_dim=args.x_dim, y_dim=args.y_dim,
                              fs=args.filter_size, mul=args.channel_mul,
                              onthefly=False,
                              use_mse_loss=args.use_mse_loss,
                              use_sigm_loss=args.use_sigm_loss)
        model.a('input_coords', input_coords)
        model.a('input_images', input_images)
        # call model on inputs
        model([input_coords, input_images])

    elif args.arch == 'coordconv_classification':
        model = CoordConvPainter(
            l2=args.l2,
            x_dim=args.x_dim,
            y_dim=args.y_dim,
            include_r=False,
            mul=args.channel_mul,
            use_mse_loss=args.use_mse_loss,
            use_sigm_loss=args.use_sigm_loss)

        model.a('input_coords', input_coords)
        model.a('input_onehot', input_onehot)

        model([input_coords, input_onehot])
        #raise Exception('Not implemented yet')

    elif args.arch == 'coordconv_rendering':
        model = CoordConvImagePainter(
            l2=args.l2,
            x_dim=args.x_dim,
            y_dim=args.y_dim,
            include_r=False,
            mul=args.channel_mul,
            fs=args.filter_size,
            use_mse_loss=args.use_mse_loss,
            use_sigm_loss=args.use_sigm_loss,
            interm_loss=args.interm_loss,
            no_softmax=args.no_softmax,
            version='working')
        # version='simple') # version='simple' to hack a 9x9 all-ones filter solution
        model.a('input_coords', input_coords)
        model.a('input_onehot', input_onehot)
        model.a('input_images', input_images)

        # always input three things to calculate relevant metrics
        model([input_coords, input_onehot, input_images])
    elif args.arch == 'deconv_bottleneck':
        model = DeconvBottleneckPainter(
            l2=args.l2,
            x_dim=args.x_dim,
            y_dim=args.y_dim,
            mul=args.channel_mul,
            fs=args.filter_size,
            use_mse_loss=args.use_mse_loss,
            use_sigm_loss=args.use_sigm_loss,
            interm_loss=args.interm_loss,
            no_softmax=args.no_softmax,
            version='working')  # version='simple' to hack a 9x9 all-ones filter solution
        model.a('input_coords', input_coords)
        model.a('input_onehot', input_onehot)
        model.a('input_images', input_images)

        # always input three things to calculate relevant metrics
        model([input_coords, input_onehot, input_images])

    elif args.arch == 'upsample_conv_coords' or args.arch == 'upsample_coordconv_coords':
        _coordconv = True if args.arch == 'upsample_coordconv_coords' else False
        model = UpsampleConvPainter(
            l2=args.l2,
            x_dim=args.x_dim,
            y_dim=args.y_dim,
            mul=args.channel_mul,
            fs=args.filter_size,
            use_mse_loss=args.use_mse_loss,
            use_sigm_loss=args.use_sigm_loss,
            coordconv=_coordconv)
        model.a('input_coords', input_coords)
        model.a('input_onehot', input_onehot)
        model([input_coords, input_onehot])

    print 'All model weights:'
    summarize_weights(model.trainable_weights)
    #print 'Model summary:'
    print 'Another model summary:'
    model.summarize_named(prefix='  ')
    print_trainable_warnings(model)

    # 2. COMPUTE GRADS AND CREATE OPTIMIZER
    # a placeholder for dynamic learning rate
    input_lr = tf.placeholder(tf.float32, shape=[])
    if args.opt == 'sgd':
        opt = tf.train.MomentumOptimizer(input_lr, args.mom)
    elif args.opt == 'rmsprop':
        opt = tf.train.RMSPropOptimizer(input_lr, momentum=args.mom)
    elif args.opt == 'adam':
        opt = tf.train.AdamOptimizer(input_lr, args.beta1, args.beta2)

    grads_and_vars = opt.compute_gradients(
        model.loss,
        model.trainable_weights,
        gate_gradients=tf.train.Optimizer.GATE_GRAPH)
    train_step = opt.apply_gradients(grads_and_vars)
    # added to train_ and param_ collections
    add_grads_and_vars_hist_summaries(grads_and_vars)

    summarize_opt(opt)
    print 'LR Policy:', lr_policy

    # add_grad_summaries(grads_and_vars)
    if not args.arch.endswith('regressor'):
        image_summaries_traintest(model.logits)

    if 'input_onehot' in model.named_keys():
        image_summaries_traintest(model.input_onehot)
    if 'input_images' in model.named_keys():
        image_summaries_traintest(model.input_images)
    if 'prob' in model.named_keys():
        image_summaries_traintest(model.prob)
    if 'center_prob' in model.named_keys():
        image_summaries_traintest(model.center_prob)
    if 'center_logits' in model.named_keys():
        image_summaries_traintest(model.center_logits)
    if 'pixelwise_prob' in model.named_keys():
        image_summaries_traintest(model.pixelwise_prob)
    if 'center_logits' in model.named_keys():
        image_summaries_traintest(model.center_logits)
    if 'sharpened_logits' in model.named_keys():
        image_summaries_traintest(model.sharpened_logits)

    # 3. OPTIONALLY SAVE OR LOAD VARIABLES (e.g. model params, model running
    # BN means, optimization momentum, ...) and then finalize initialization
    saver = tf.train.Saver(
        max_to_keep=None) if (
        args.output or args.load) else None
    if args.load:
        ckptfile, miscfile = args.load.split(':')
        # Restore values directly to graph
        saver.restore(sess, ckptfile)
        with gzip.open(miscfile) as ff:
            saved = pickle.load(ff)
            buddy = saved['buddy']
    else:
        buddy = StatsBuddy()

    buddy.tic()    # call if new run OR resumed run

    # Check if special layers are initialized right
    #last_layer_w = [var for var in tf.global_variables() if 'painting_layer/kernel:0' in var.name][0]
    #last_layer_b = [var for var in tf.global_variables() if 'painting_layer/bias:0' in var.name][0]

    # Initialize any missed vars (e.g. optimization momentum, ... if not
    # loaded from checkpoint)
    uninitialized_vars = tf_get_uninitialized_variables(sess)
    init_missed_vars = tf.variables_initializer(
        uninitialized_vars, 'init_missed_vars')
    sess.run(init_missed_vars)
    # Print warnings about any TF vs. Keras shape mismatches
    # warn_misaligned_shapes(model)
    # Make sure all variables, which are model variables, have been
    # initialized (e.g. model params and model running BN means)
    tf_assert_all_init(sess)
    # tf.global_variables_initializer().run()

    # 4. SETUP TENSORBOARD LOGGING with tf.summary.merge

    train_histogram_summaries = get_collection_intersection_summary(
        'train_collection', 'orig_histogram')
    train_scalar_summaries = get_collection_intersection_summary(
        'train_collection', 'orig_scalar')
    test_histogram_summaries = get_collection_intersection_summary(
        'test_collection', 'orig_histogram')
    test_scalar_summaries = get_collection_intersection_summary(
        'test_collection', 'orig_scalar')
    param_histogram_summaries = get_collection_intersection_summary(
        'param_collection', 'orig_histogram')
    train_image_summaries = get_collection_intersection_summary(
        'train_collection', 'orig_image')
    test_image_summaries = get_collection_intersection_summary(
        'test_collection', 'orig_image')

    writer = None
    if args.output:
        mkdir_p(args.output)
        writer = tf.summary.FileWriter(args.output, sess.graph)

    # 5. TRAIN

    train_iters = (train_size) // minibatch_size + \
        int(train_size % minibatch_size > 0)
    if not args.skipval:
        val_iters = (val_size) // minibatch_size + \
            int(val_size % minibatch_size > 0)

    if args.ipy:
        print 'Embed: before train / val loop (Ctrl-D to continue)'
        embed()

    while buddy.epoch < args.epochs + 1:
        # How often to log data
        def do_log_params(ep, it, ii): return True
        def do_log_val(ep, it, ii): return True

        def do_log_train(
            ep,
            it,
            ii): return (
            it < train_iters and it & it -
            1 == 0 or it >= train_iters and it %
            train_iters == 0)  # Log on powers of two then every epoch

        # 0. Log params
        if args.output and do_log_params(
                buddy.epoch,
                buddy.train_iter,
                0) and param_histogram_summaries is not None:
            params_summary_str, = sess.run([param_histogram_summaries])
            writer.add_summary(params_summary_str, buddy.train_iter)

        # 1. Forward test on validation set
        if not args.skipval:
            feed_dict = {learning_phase(): 0}
            if 'input_coords' in model.named_keys():
                val_coords = val_y if args.arch.endswith(
                    'regressor') else val_x
                feed_dict.update({model.input_coords: val_coords})

            if 'input_onehot' in model.named_keys():
                # if 'val_onehot' not in locals():
                if not args.arch == 'coordconv_rendering' and not args.arch == 'deconv_bottleneck':
                    if args.arch == 'conv_onehot_image' or args.arch.endswith('regressor'):
                        val_onehot = val_x
                    else:
                        val_onehot = val_y
                feed_dict.update({
                    model.input_onehot: val_onehot,
                })
            if 'input_images' in model.named_keys():
                feed_dict.update({
                    model.input_images: val_images,
                })

            fetch_dict = model.trackable_dict()

            if args.output and do_log_val(buddy.epoch, buddy.train_iter, 0):
                if test_image_summaries is not None:
                    fetch_dict.update(
                        {'test_image_summaries': test_image_summaries})
                if test_scalar_summaries is not None:
                    fetch_dict.update(
                        {'test_scalar_summaries': test_scalar_summaries})
                if test_histogram_summaries is not None:
                    fetch_dict.update(
                        {'test_histogram_summaries': test_histogram_summaries})

            with WithTimer('sess.run val iter', quiet=not args.verbose):
                result_val = sess_run_dict(
                    sess, fetch_dict, feed_dict=feed_dict)

            buddy.note_list(
                model.trackable_names(), [
                    result_val[k] for k in model.trackable_names()], prefix='val_')
            print (
                '[%5d] [%2d/%2d] val: %s (%.3gs/i)' %
                (buddy.train_iter,
                 buddy.epoch,
                 args.epochs,
                 buddy.epoch_mean_pretty_re(
                     '^val_',
                     style=val_style),
                    toc2()))

            if args.output and do_log_val(buddy.epoch, buddy.train_iter, 0):
                log_scalars(
                    writer, buddy.train_iter, {
                        'mean_%s' %
                        name: value for name, value in buddy.epoch_mean_list_re('^val_')}, prefix='val')
                if test_image_summaries is not None:
                    image_summary_str = result_val['test_image_summaries']
                    writer.add_summary(image_summary_str, buddy.train_iter)
                if test_scalar_summaries is not None:
                    scalar_summary_str = result_val['test_scalar_summaries']
                    writer.add_summary(scalar_summary_str, buddy.train_iter)
                if test_histogram_summaries is not None:
                    hist_summary_str = result_val['test_histogram_summaries']
                    writer.add_summary(hist_summary_str, buddy.train_iter)

        # 2. Possiby Snapshot, possibly quit
        if args.output and args.snapshot_to and args.snapshot_every:
            snap_intermed = args.snapshot_every > 0 and buddy.train_iter % args.snapshot_every == 0
            #snap_end = buddy.epoch == args.epochs
            snap_end = lr_policy.train_done(buddy)
            if snap_intermed or snap_end:
                # Snapshot network and buddy
                save_path = saver.save(
                    sess, '%s/%s_%04d.ckpt' %
                    (args.output, args.snapshot_to, buddy.epoch))
                print 'snappshotted model to', save_path
                with gzip.open('%s/%s_misc_%04d.pkl.gz' % (args.output, args.snapshot_to, buddy.epoch), 'w') as ff:
                    saved = {'buddy': buddy}
                    pickle.dump(saved, ff)
                # Snapshot evaluation data and metrics
                _, _ = evaluate_net(
                    args, buddy, model, train_size, train_x, train_y, val_x, val_y, fd, sess)

        lr = lr_policy.get_lr(buddy)

        if buddy.epoch == args.epochs:
            if args.ipy:
                print 'Embed: at end of training (Ctrl-D to exit)'
                embed()
            break   # Extra pass at end: just report val stats and skip training

        print '********* at epoch %d, LR is %g' % (buddy.epoch, lr)

        # 3. Train on training set
        if args.shuffletrain:
            train_order = np.random.permutation(train_size)
        tic3()
        for ii in xrange(train_iters):
            tic2()
            start_idx = ii * minibatch_size
            end_idx = min(start_idx + minibatch_size, train_size)

            if args.shuffletrain:  # default true
                batch_x = train_x[sorted(
                    train_order[start_idx:end_idx].tolist())]
                if train_y is not None:
                    batch_y = train_y[sorted(
                        train_order[start_idx:end_idx].tolist())]
                # if 'train_onehot' in locals():
                if args.arch == 'coordconv_rendering' or args.arch == 'deconv_bottleneck':
                    batch_onehot = train_onehot[sorted(
                        train_order[start_idx:end_idx].tolist())]
            else:
                batch_x = train_x[start_idx:end_idx]
                if train_y is not None:
                    batch_y = train_y[start_idx:end_idx]
                # if 'train_onehot' in locals():
                if args.arch == 'coordconv_rendering' or args.arch == 'deconv_bottleneck':
                    batch_onehot = train_onehot[start_idx:end_idx]

            feed_dict = {learning_phase(): 1, input_lr: lr}
            if 'input_coords' in model.named_keys():
                batch_coords = batch_y if args.arch.endswith(
                    'regressor') else batch_x
                feed_dict.update({model.input_coords: batch_coords})
            if 'input_onehot' in model.named_keys():
                # if 'batch_onehot' not in locals():
                # if not (args.arch == 'coordconv_rendering' and
                # args.add_interm_loss):
                if not args.arch == 'coordconv_rendering' and not args.arch == 'deconv_bottleneck':
                    if args.arch == 'conv_onehot_image' or args.arch.endswith(
                            'regressor'):
                        batch_onehot = batch_x
                    else:
                        batch_onehot = batch_y
                feed_dict.update({
                    model.input_onehot: batch_onehot,
                })
            if 'input_images' in model.named_keys():
                feed_dict.update({
                    model.input_images: batch_images,
                })

            fetch_dict = model.trackable_and_update_dict()

            fetch_dict.update({'train_step': train_step})

            if args.output and do_log_train(buddy.epoch, buddy.train_iter, ii):
                if train_histogram_summaries is not None:
                    fetch_dict.update(
                        {'train_histogram_summaries': train_histogram_summaries})
                if train_scalar_summaries is not None:
                    fetch_dict.update(
                        {'train_scalar_summaries': train_scalar_summaries})
                if train_image_summaries is not None:
                    fetch_dict.update(
                        {'train_image_summaries': train_image_summaries})

            with WithTimer('sess.run train iter', quiet=not args.verbose):
                result_train = sess_run_dict(
                    sess, fetch_dict, feed_dict=feed_dict)

            buddy.note_weighted_list(
                batch_x.shape[0], model.trackable_names(), [
                    result_train[k] for k in model.trackable_names()], prefix='train_')

            if do_log_train(buddy.epoch, buddy.train_iter, ii):
                print (
                    '[%5d] [%2d/%2d] train: %s (%.3gs/i)' %
                    (buddy.train_iter,
                     buddy.epoch,
                     args.epochs,
                     buddy.epoch_mean_pretty_re(
                         '^train_',
                         style=train_style),
                        toc2()))

            if args.output and do_log_train(buddy.epoch, buddy.train_iter, ii):
                if train_histogram_summaries is not None:
                    hist_summary_str = result_train['train_histogram_summaries']
                    writer.add_summary(hist_summary_str, buddy.train_iter)
                if train_scalar_summaries is not None:
                    scalar_summary_str = result_train['train_scalar_summaries']
                    writer.add_summary(scalar_summary_str, buddy.train_iter)
                if train_image_summaries is not None:
                    image_summary_str = result_train['train_image_summaries']
                    writer.add_summary(image_summary_str, buddy.train_iter)
                log_scalars(
                    writer, buddy.train_iter, {
                        'batch_%s' %
                        name: value for name, value in buddy.last_list_re('^train_')}, prefix='train')

            if ii > 0 and ii % 100 == 0:
                print '  %d: Average iteration time over last 100 train iters: %.3gs' % (
                    ii, toc3() / 100)
                tic3()

            buddy.inc_train_iter()   # after finished training a mini-batch

        buddy.inc_epoch()   # after finished training whole pass through set

        if args.output and do_log_train(buddy.epoch, buddy.train_iter, 0):
            log_scalars(
                writer, buddy.train_iter, {
                    'mean_%s' %
                    name: value for name, value in buddy.epoch_mean_list_re('^train_')}, prefix='train')

    print '\nFinal'
    print '%02d:%d val:   %s' % (buddy.epoch,
                                 buddy.train_iter,
                                 buddy.epoch_mean_pretty_re(
                                     '^val_',
                                     style=val_style))
    print '%02d:%d train: %s' % (buddy.epoch,
                                 buddy.train_iter,
                                 buddy.epoch_mean_pretty_re(
                                     '^train_',
                                     style=train_style))

    print '\nEnd of training. Saving evaluation results on whole train and val set.'

    final_tr_metrics, final_va_metrics = evaluate_net(
        args, buddy, model, train_size, train_x, train_y, val_x, val_y, fd, sess)

    print '\nFinal evaluation on whole train and val'
    for name, value in final_tr_metrics.iteritems():
        print 'final_stats_eval train_%s %g' % (name, value)
    for name, value in final_va_metrics.iteritems():
        print 'final_stats_eval val_%s %g' % (name, value)

    print '\nfinal_stats epochs %g' % buddy.epoch
    print 'final_stats iters %g' % buddy.train_iter
    print 'final_stats time %g' % buddy.toc()
    for name, value in buddy.epoch_mean_list_all():
        print 'final_stats %s %g' % (name, value)

    if args.output:
        writer.close()   # Flush and close


def evaluate_net(args, buddy, model, train_size, train_x, train_y,
                 val_x, val_y, fd, sess, write_x=True, write_y=True):

    minibatch_size = args.minibatch
    train_iters = (train_size) // minibatch_size + \
        int(train_size % minibatch_size > 0)

    # 0 even for train set; because it's evalutation
    feed_dict_tr = {learning_phase(): 0}
    feed_dict_va = {learning_phase(): 0}

    if args.output:
        final_fetch = {'logits': model.logits}
        if 'prob' in model.named_keys():
            final_fetch.update({'prob': model.prob})
        if 'pixelwise_prob' in model.named_keys():
            final_fetch.update({'pixelwise_prob': model.pixelwise_prob})

        if args.arch == 'coordconv_rendering' or args.arch == 'deconv_bottleneck':
            final_fetch.update({
                'center_logits': model.center_logits,
                # 'sharpened_logits': model.sharpened_logits, # or center_prob
                'center_prob': model.center_prob,  # or center_prob
            })

        ff = h5py.File(
            '%s/evaluation_%04d.h5' %
            (args.output, buddy.epoch), 'w')

        # create dataset but write later
        for kk in final_fetch.keys():
            if args.arch.endswith('regressor'):
                ff.create_dataset(kk + '_train', (minibatch_size, 2),
                    maxshape=(train_size, 2), dtype=float, 
                    compression='lzf', chunks=True)
            else:
                ff.create_dataset(kk + '_train',
                    (minibatch_size, args.x_dim, args.y_dim, 1),
                    maxshape=(train_size, args.x_dim, args.y_dim, 1),
                    dtype=float, compression='lzf', chunks=True)

        # create dataset and write immediately
        if write_x:
            ff.create_dataset('inputs_val', data=val_x)
            ff.create_dataset('inputs_train', data=train_x)
        if write_y:
            ff.create_dataset('labels_val', data=val_y)
            ff.create_dataset('labels_train', data=train_y)

    for ii in xrange(train_iters):
        start_idx = ii * minibatch_size
        end_idx = min(start_idx + minibatch_size, train_size)

        if 'input_onehot' in model.named_keys():
            feed_dict_tr.update({model.input_onehot: np.array(
                fd['train_onehots'][start_idx:end_idx], dtype=float)})
            if ii == 0:
                feed_dict_va.update(
                    {model.input_onehot: np.array(fd['val_onehots'], dtype=float)})
                #feed_dict_va.update({model.input_onehot: val_onehot})
        if 'input_images' in model.named_keys():
            feed_dict_tr.update({model.input_images: np.array(
                fd['train_imagegray'][start_idx:end_idx], dtype=float) / 255.0})
            if ii == 0:
                feed_dict_va.update({model.input_images: np.array(
                    fd['val_imagegray'], dtype=float) / 255.0})
                #feed_dict_va.update({model.input_images: val_images})

        if 'input_coords' in model.named_keys():
            if args.arch.endswith('regressor'):
                _loc_keys = (
                    'train_normalized_locations',
                    'val_normalized_locations',
                    'float32')
            else:
                _loc_keys = (
                    'train_locations', 
                    'val_locations', 
                    'int32')
            feed_dict_tr.update({model.input_coords: np.array(
                fd[_loc_keys[0]][start_idx:end_idx], dtype=_loc_keys[2])})
            if ii == 0:
                feed_dict_va.update({model.input_coords: np.array(
                    fd[_loc_keys[1]], dtype=_loc_keys[2])})

        _final_tr_metrics = sess_run_dict(
            sess, model.trackable_dict(), feed_dict=feed_dict_tr)
        _final_tr_metrics['weights'] = end_idx - start_idx

        final_tr_metrics = _final_tr_metrics if ii == 0 else merge_dict_append(
            final_tr_metrics, _final_tr_metrics)

        if args.output:
            if ii == 0:  # do only once
                final_va = sess_run_dict(
                    sess, final_fetch, feed_dict=feed_dict_va)
                for kk in final_fetch.keys():
                    ff.create_dataset(kk + '_val', data=final_va[kk])

            final_tr = sess_run_dict(sess, final_fetch, feed_dict=feed_dict_tr)
            for kk in final_fetch.keys():
                if start_idx > 0:
                    n_samples_ = ff[kk + '_train'].shape[0]
                    ff[kk + '_train'].resize(n_samples_ +
                                             end_idx - start_idx, axis=0)
                ff[kk + '_train'][start_idx:, ...] = final_tr[kk]

    final_va_metrics = sess_run_dict(
        sess, model.trackable_dict(), feed_dict=feed_dict_va)
    final_tr_metrics = average_dict_values(final_tr_metrics)

    if args.output:
        with open('%s/evaluation_%04d_metrics.pkl' % (args.output, buddy.epoch), 'w') as ffmetrics:
            tosave = {'train': final_tr_metrics,
                      'val': final_va_metrics,
                      'time_elapsed': buddy.toc()
                      }
            pickle.dump(tosave, ffmetrics)

        ff.close()
    else:
        print '\nEpoch %d evaluation on whole train and val' % buddy.epoch
        print 'Time elapsed: {}'.format(buddy.toc())
        for name, value in final_tr_metrics.iteritems():
            print 'final_stats_eval train_%s %g' % (name, value)
        for name, value in final_va_metrics.iteritems():
            print 'final_stats_eval val_%s %g' % (name, value)

    return final_tr_metrics, final_va_metrics


if __name__ == '__main__':
    main()
