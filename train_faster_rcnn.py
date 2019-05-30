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
import _pickle as pickle
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
from util import make_standard_parser, merge_dict_append, average_dict_values

from model_builder_faster_rcnn import RegionProposalSampler
from utils_faster_rcnn import make_anchors_mnist, make_anchors_mnist_same, plot_pos_boxes, plot_boxes_pos_neg, plot_pos_boxes_thickness
from field_of_mnist.loader import load_tvt_n_per_field, load_tvt_n_per_field_centercrop
from params import RPNParams, BoxSamplerParams, NMSParams
from matplotlib.pyplot import *

# args.arch is one of the elements in arch_choices
arch_choices = ['rpn_sampler', 'coord_rpn_sampler', 'rpn_sampler_stn']

def main():
    lr_policy_choices = ('constant', 'step', 'valstep')
    
    parser = make_standard_parser('Region Proposal Net',
                                  arch_choices=arch_choices, skip_train=True, skip_val=True)
    parser.add_argument('--num', '-N', type=int, default=2, 
                        help='Load the Field-of-MNIST dataset with NUM digits per image.')
    parser.add_argument('--lrpolicy', type=str, default='constant', 
                        choices=lr_policy_choices, help='LR policy.')
    parser.add_argument('--lrstepratio', type=float, 
                        default=.1, help='LR policy step ratio.')
    parser.add_argument('--lrmaxsteps', type=int, default=5, 
                        help='LR policy step ratio.')
    parser.add_argument('--lrstepevery', type=int, default=50, 
                        help='LR policy step ratio.')
    parser.add_argument('--clip', action='store_true', 
                        help='clip predicted and ground truth boxes.' )
    parser.add_argument('--same', action='store_true', 
                        help='Use `same` filter instead of `valid` in conv.' )
    parser.add_argument('--showbox', action='store_true', 
                        help='show moved box during training.' )

    args = parser.parse_args()

    if args.lrpolicy == 'step':
        lr_policy = LRPolicyStep(args)
    elif args.lrpolicy == 'valstep':
        lr_policy = LRPolicyValStep(args)
    else:
        lr_policy = LRPolicyConstant(args)

    minibatch_size = 1
    train_style, val_style = ('', '') if args.nocolor else (colorama.Fore.BLUE, colorama.Fore.MAGENTA)

    sess = setup_session_and_seeds(args.seed, assert_gpu=not args.cpu)

    # 0. Load data 
    #train_ims, train_pos, train_class, valid_ims, valid_pos, valid_class, _, _, _ = load_tvt_n_per_field(args.num)
    #train_ims = train_ims[:5000]     # (5000, 64, 64, 1)
    #train_pos = train_pos[:5000]     # (5000, 2, 4)
    #valid_ims = valid_ims[:1000]     
    #valid_pos = valid_pos[:1000]
    #_ims, _pos, _class, _, _, _, _, _, _ = load_tvt_n_per_field_centercrop(args.num)
    
    ff = h5py.File('data/field_of_mnist_cropped_64x64_5objs.h5', 'r')
    
    train_ims = np.array(ff['train_ims'])  # (9000, 64, 64, 1)
    train_pos = np.array(ff['train_pos'])  # (9000, 5, 4), parts of boxes may be out of canvas
    train_class = np.array(ff['train_class']) # (9000, 5)
    valid_ims = np.array(ff['valid_ims'])  # (1000, 64, 64, 1)
    valid_pos = np.array(ff['valid_pos'])  # (1000, 5, 4), parts of boxes may be out of canvas
    valid_class = np.array(ff['valid_class']) # (1000, 5)

    ff.close()
    

    im_h, im_w, im_c = train_ims.shape[1], train_ims.shape[2], train_ims.shape[3]
    train_size = train_ims.shape[0]
    val_size = valid_ims.shape[0]
    
    print(('Data loaded:\n\timage shape: {}x{}x{}'.format(im_h, im_w, im_c)))
    print(('\ttrain size: {}\n\ttest size: {}'.format(train_size, val_size)))
    print(('\tnumber of objects per image: {}'.format(train_pos.shape[1])))

    ####################
    # RPN prameters
    ####################
    rpn_params = RPNParams(
        anchors=np.array([
            (15, 15), (20, 20), (25, 25),
            (15, 20), (20, 25), (20, 15),
            (25, 20), (15, 25), (25, 15)]),
        rpn_hidden_dim=32,
        zero_box_conv=False,
        weight_init_std=0.01,
        anchor_scale=1.0)

    bsamp_params = BoxSamplerParams(
        hi_thresh=0.5,
        lo_thresh=0.1,
        sample_size=12)

    nms_params = NMSParams(
        nms_thresh = 0.8,
        max_proposals = 10,
    )

    # 1. CREATE MODEL

    input_images = tf.placeholder(shape=(None, im_h, im_w, im_c), dtype='float32', name='input_images') 
    input_gtbox = tf.placeholder(shape=(train_pos.shape[1],4), dtype='float32', name='input_gtbox')
    
    if args.arch == 'rpn_sampler':
        model = RegionProposalSampler(rpn_params, bsamp_params, nms_params, l2=args.l2,
                                    im_h=im_h, im_w=im_w, coordconv=False, 
                                    clip=args.clip, filtersame=args.same)
    elif args.arch == 'coord_rpn_sampler':
        model = RegionProposalSampler(rpn_params, bsamp_params, nms_params, l2=args.l2,
                                    im_h=im_h, im_w=im_w, coordconv=True,
                                    clip=args.clip, filtersame=args.same)
    else:
        raise ValueError('Architecture {} unknown'.format(args.arch))
    
    if args.same:
        anchors = make_anchors_mnist_same((16,16), minibatch_size, rpn_params.anchors)     # (batch, 16, 16, 4k)
        input_anchors = tf.placeholder(shape=(16,16,4*rpn_params.num_anchors), dtype='float32', name='input_anchors')
    else:
        anchors = make_anchors_mnist((13,13), minibatch_size, rpn_params.anchors)     # (batch, 13, 13, 4k)
        input_anchors = tf.placeholder(shape=(13,13,4*rpn_params.num_anchors), dtype='float32', name='input_anchors')
    anchors = anchors[0]

    model.a('input_images', input_images)
    model.a('input_anchors', input_anchors)
    model.a('input_gtbox', input_gtbox)
       
    model([input_images, input_anchors, input_gtbox])

    print('All model weights:')
    summarize_weights(model.trainable_weights)
    #print 'Model summary:'
    print('Another model summary:')
    model.summarize_named(prefix='  ')
    print_trainable_warnings(model)

    # 2. COMPUTE GRADS AND CREATE OPTIMIZER
    input_lr = tf.placeholder(tf.float32, shape=[])  # a placeholder for dynamic learning rate
    if args.opt == 'sgd':
        opt = tf.train.MomentumOptimizer(input_lr, args.mom)
    elif args.opt == 'rmsprop':
        opt = tf.train.RMSPropOptimizer(input_lr, momentum=args.mom)
    elif args.opt == 'adam':
        opt = tf.train.AdamOptimizer(input_lr, args.beta1, args.beta2)

    grads_and_vars = opt.compute_gradients(model.loss, model.trainable_weights, gate_gradients=tf.train.Optimizer.GATE_GRAPH)
    train_step = opt.apply_gradients(grads_and_vars)
    add_grads_and_vars_hist_summaries(grads_and_vars) # added to train_ and param_ collections

    summarize_opt(opt)
    print(('LR Policy:', lr_policy))

    #add_grad_summaries(grads_and_vars)
    #image_summaries_traintest(model.logits)
    #if 'input_1hot' in model.named_keys(): 
    #    image_summaries_traintest(model.input_1hot)
    #if 'input_images' in model.named_keys(): 
    #    image_summaries_traintest(model.input_images)
    #if 'prob' in model.named_keys(): 
    #    image_summaries_traintest(model.prob)
    #if 'center_prob' in model.named_keys(): 
    #    image_summaries_traintest(model.center_prob)
    #if 'center_logits' in model.named_keys(): 
    #    image_summaries_traintest(model.center_logits)
    #if 'pixelwise_prob' in model.named_keys(): 
    #    image_summaries_traintest(model.pixelwise_prob)
    #if 'center_logits' in model.named_keys(): 
    #    image_summaries_traintest(model.center_logits)
    #if 'sharpened_logits' in model.named_keys(): 
    #    image_summaries_traintest(model.sharpened_logits)


    # 3. OPTIONALLY SAVE OR LOAD VARIABLES (e.g. model params, model running BN means, optimization momentum, ...) and then finalize initialization
    saver = tf.train.Saver(max_to_keep=None) if (args.output or args.load) else None
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

    # Initialize any missed vars (e.g. optimization momentum, ... if not loaded from checkpoint)
    uninitialized_vars = tf_get_uninitialized_variables(sess)
    init_missed_vars = tf.variables_initializer(uninitialized_vars, 'init_missed_vars')
    sess.run(init_missed_vars)
    ## Print warnings about any TF vs. Keras shape mismatches
    ##warn_misaligned_shapes(model)
    ## Make sure all variables, which are model variables, have been initialized (e.g. model params and model running BN means)
    tf_assert_all_init(sess)
    #tf.global_variables_initializer().run()


    # 4. SETUP TENSORBOARD LOGGING with tf.summary.merge

    train_histogram_summaries = get_collection_intersection_summary('train_collection', 'orig_histogram')
    train_scalar_summaries    = get_collection_intersection_summary('train_collection', 'orig_scalar')
    test_histogram_summaries   = get_collection_intersection_summary('test_collection', 'orig_histogram')
    test_scalar_summaries      = get_collection_intersection_summary('test_collection', 'orig_scalar')
    param_histogram_summaries = get_collection_intersection_summary('param_collection', 'orig_histogram')
    train_image_summaries     = get_collection_intersection_summary('train_collection', 'orig_image')
    test_image_summaries     = get_collection_intersection_summary('test_collection', 'orig_image')

    writer = None
    if args.output:
        mkdir_p(args.output)
        writer = tf.summary.FileWriter(args.output, sess.graph)

    # 5. TRAIN

    train_iters = (train_size) // minibatch_size 
    if not args.skipval:
        val_iters = (val_size) // minibatch_size
    
    if args.output:
        show_indices = np.random.permutation(val_size)[:9]
        mkdir_p('{}/figures'.format(args.output))

    if args.ipy:
        print('Embed: before train / val loop (Ctrl-D to continue)')
        embed()

    while buddy.epoch < args.epochs + 1:
        # How often to log data
        do_log_params = lambda ep, it, ii: True
        do_log_val = lambda ep, it, ii: True
        do_log_train = lambda ep, it, ii: (it < train_iters and it & it-1 == 0 or it>=train_iters and it%train_iters == 0)  # Log on powers of two then every epoch

        # 0. Log params
        if args.output and do_log_params(buddy.epoch, buddy.train_iter, 0) and param_histogram_summaries is not None:
            params_summary_str, = sess.run([param_histogram_summaries])
            writer.add_summary(params_summary_str, buddy.train_iter)


        # 1. Forward test on validation set
        if not args.skipval:
            for ii in range(val_iters):
                tic2()
                start_idx = ii * minibatch_size
                end_idx = min(start_idx + minibatch_size, val_size)
                if not end_idx > start_idx:
                    continue
            
                feed_dict = {
                        model.input_images: valid_ims[start_idx:end_idx],
                        model.input_anchors: anchors,
                        model.input_gtbox: valid_pos[start_idx:end_idx][0],
                        learning_phase(): 0}
                
                fetch_dict = model.trackable_dict()
                    
                if args.output and do_log_val(buddy.epoch, buddy.train_iter, 0): 
                    if test_image_summaries is not None:
                        fetch_dict.update({'test_image_summaries':test_image_summaries})
                    if test_scalar_summaries is not None:
                        fetch_dict.update({'test_scalar_summaries':test_scalar_summaries})
                    if test_histogram_summaries is not None:
                        fetch_dict.update({'test_histogram_summaries':test_histogram_summaries})

                with WithTimer('sess.run val iter', quiet=not args.verbose):
                    result_val = sess_run_dict(sess, fetch_dict, feed_dict=feed_dict)


                ## DEBUG
                ## dynamic p_size and n_size, shouldn slightly very every sample 
                #if ii > 0 and ii % 100 == 0:
                #    print 'VALIDATION --- '
                #    print sess.run(model.p_size, feed_dict=feed_dict)
                #    print sess.run(model.n_size, feed_dict=feed_dict)
                ## END DEBUG

                buddy.note_weighted_list(minibatch_size, model.trackable_names(), [result_val[k] for k in model.trackable_names()], prefix='val_')
            
            # Done all val set
            print(('[%5d] [%2d/%2d] val: %s (%.3gs/i)' % (buddy.train_iter, buddy.epoch, args.epochs, buddy.epoch_mean_pretty_re('^val_', style=val_style), toc2())))
            
            
            if args.output and do_log_val(buddy.epoch, buddy.train_iter, 0):
                log_scalars(writer, buddy.train_iter,
                            {'mean_%s' % name: value for name, value in buddy.epoch_mean_list_re('^val_')},
                            prefix='val')
                if test_image_summaries is not None:
                    image_summary_str = result_val['test_image_summaries']
                    writer.add_summary(image_summary_str, buddy.train_iter)
                if test_scalar_summaries is not None:
                    scalar_summary_str = result_val['test_scalar_summaries']
                    writer.add_summary(scalar_summary_str, buddy.train_iter)
                if test_histogram_summaries is not None:
                    hist_summary_str = result_val['test_histogram_summaries']
                    writer.add_summary(hist_summary_str, buddy.train_iter)

        # show some boxes
        if args.showbox: #and (valid_losses[epoch]/previous_best < 1- args.thresh):
            show_indices = [55, 555, 678]
            for show_idx in show_indices:
                [pos_box, pos_score, neg_box, neg_score] = sess.run([model.pos_box, model.pos_score, 
                                                                     model.neg_box, model.neg_score], 
                                                                feed_dict={
                                                                    model.input_images: valid_ims[show_idx:show_idx+1],
                                                                    model.input_anchors: anchors,
                                                                    model.input_gtbox: valid_pos[show_idx],
                                                                    learning_phase(): 0}
                                                                    )
                subplot(1,3,show_indices.index(show_idx)+1)
                #plot_boxes_pos_neg(valid_ims[show_idx], valid_pos[show_idx], pos_box, neg_box)
                plot_pos_boxes(valid_ims[show_idx], valid_pos[show_idx], pos_box, pos_score, showlabel=False)
            show()

        if args.output:
            switch_backend('Agg')
            plot_fetch_dict = {
                    'pos_box': model.pos_box, 
                    'pos_score': model.pos_score, 
                    'neg_box': model.neg_box, 
                    'neg_score': model.neg_score,
                    'nms_boxes': model.nms_boxes,
                    'nms_scores': model.nms_scores,
                    }
            
            #fig1, ax1 = subplots(3,3)  # plot train boxes
            #fig2, ax2 = subplots(3,3)  # plot test/nms boxes
            for cc, show_idx in enumerate(show_indices, 1):
                feed_dict={
                    model.input_images: valid_ims[show_idx:show_idx+1],
                    model.input_anchors: anchors,
                    model.input_gtbox: valid_pos[show_idx],
                    learning_phase(): 0}
                result_plots = sess_run_dict(sess, plot_fetch_dict, feed_dict=feed_dict)
                fig1 = figure(1)
                subplot(3,3,cc)
                plot_boxes_pos_neg(valid_ims[show_idx], valid_pos[show_idx], result_plots['pos_box'], result_plots['neg_box'])
                fig2 = figure(2)
                subplot(3,3,cc)
                #plot_pos_boxes(valid_ims[show_idx], valid_pos[show_idx], result_plots['nms_boxes'], result_plots['nms_scores'], showlabel=False)
                # normalize scores between 0 and 5, to be used as line width
                _score_as_lw = 5 * (result_plots['nms_scores'] - result_plots['nms_scores'].min()) / (result_plots['nms_scores'].max() - result_plots['nms_scores'].min())
                plot_pos_boxes_thickness(valid_ims[show_idx], valid_pos[show_idx], result_plots['nms_boxes'], result_plots['nms_scores'])

            fig1.set_size_inches(10, 10)
            fig1.savefig('{}/figures/pos_neg_train_box_epoch_{}.png'.format(args.output, buddy.epoch), dpi=100)
            fig2.set_size_inches(10, 10)
            fig2.savefig('{}/figures/nms_test_box_epoch_{}.png'.format(args.output, buddy.epoch), dpi=100)

            # plot test/nms boxes
            fig, _ = subplots()






        # 2. Possiby Snapshot, possibly quit
        if args.output and args.snapshot_to and args.snapshot_every:
            snap_intermed = args.snapshot_every > 0 and buddy.train_iter % args.snapshot_every == 0
            #snap_end = buddy.epoch == args.epochs
            snap_end = lr_policy.train_done(buddy)
            if snap_intermed or snap_end:
                # Snapshot network and buddy
                save_path = saver.save(sess, '%s/%s_%04d.ckpt' % (args.output, args.snapshot_to, buddy.epoch))
                print(('snappshotted model to', save_path))
                with gzip.open('%s/%s_misc_%04d.pkl.gz' % (args.output, args.snapshot_to, buddy.epoch), 'w') as ff:
                    saved = {'buddy': buddy}
                    pickle.dump(saved, ff)

        lr = lr_policy.get_lr(buddy)

        if buddy.epoch == args.epochs:
            if args.ipy:
                print('Embed: at end of training (Ctrl-D to exit)')
                embed()
            break   # Extra pass at end: just report val stats and skip training
        
        print(('********* at epoch %d, LR is %g' % (buddy.epoch, lr)))

        # 3. Train on training set
        if args.shuffletrain:
            train_order = np.random.permutation(train_size)
        tic3()
        for ii in range(train_iters):
            tic2()
            start_idx = ii * minibatch_size
            end_idx = min(start_idx + minibatch_size, train_size)

            if not end_idx > start_idx:
                continue

            if args.shuffletrain: # default true
                batch_ims = train_ims[sorted(train_order[start_idx:end_idx].tolist())]
                batch_pos = train_pos[sorted(train_order[start_idx:end_idx].tolist())]
            else:
                batch_ims = train_ims[start_idx:end_idx]
                batch_pos = train_pos[start_idx:end_idx]
                            
            feed_dict = {
                        model.input_images: batch_ims,
                        model.input_anchors: anchors,
                        model.input_gtbox: batch_pos[0],
                        learning_phase(): 1, 
                        input_lr: lr}

            fetch_dict = model.trackable_and_update_dict()

            fetch_dict.update({'train_step': train_step})
            
            if args.output and do_log_train(buddy.epoch, buddy.train_iter, ii):
                if train_histogram_summaries is not None:
                    fetch_dict.update({'train_histogram_summaries': train_histogram_summaries})
                if train_scalar_summaries is not None:
                    fetch_dict.update({'train_scalar_summaries': train_scalar_summaries})
                if train_image_summaries is not None:
                    fetch_dict.update({'train_image_summaries': train_image_summaries})
                
            with WithTimer('sess.run train iter', quiet=not args.verbose):
                result_train = sess_run_dict(sess, fetch_dict, feed_dict=feed_dict)
                                
            #if ii == 0:
            #if ii== 0 and result_train['mse_loss'] < 0.01:
            #    plot_fetch_dict = {
            #            'prob': model.prob,
            #            'prob_flat': model.prob_flat,
            #            'logits': model.logits,
            #            'logits_flat': model.logits_flat,
            #            'labels': model.labels,
            #            'labels_flat': model.labels_flat,
            #            #'concat_indices': model.concat_indices,
            #            'argmax_prob': model.argmax_prob,
            #            'argmax_label': model.argmax_label,
            #            'argmax_x': model.argmax_x,
            #            'argmax_y': model.argmax_y,
            #            'argmax_x_l': model.argmax_x_l,
            #            'argmax_y_l': model.argmax_y_l,
            #            }
            #    results = sess_run_dict(sess, plot_fetch_dict, feed_dict=feed_dict)
            #    embed()
            
            #if ii == 0:
            #    plot_fetch_dict = {
            #            'pixelwise_prob_flat': model.pixelwise_prob_flat,
            #            'logits_flat': model.logits_flat,
            #            'labels_flat': model.labels_flat,
            #            'painted': model.painted,
            #            'n_intersection': model.n_intersection,
            #            'n_union': model.n_union,
            #            }
            #    results = sess_run_dict(sess, plot_fetch_dict, feed_dict=feed_dict)
            #    embed()
            #    HERE

            #if ii == 0:
            #    plot_fetch_dict = {
            #            'prepped_coords': model.prepped_coords,
            #            }
            #    results = sess_run_dict(sess, plot_fetch_dict, feed_dict=feed_dict)
            #    embed()
            #    HERE

            buddy.note_weighted_list(minibatch_size, model.trackable_names(), [result_train[k] for k in model.trackable_names()], prefix='train_')

            if do_log_train(buddy.epoch, buddy.train_iter, ii):
                print(('[%5d] [%2d/%2d] train: %s (%.3gs/i)' % (buddy.train_iter, buddy.epoch, args.epochs, buddy.epoch_mean_pretty_re('^train_', style=train_style), toc2())))

            
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
                log_scalars(writer, buddy.train_iter,
                            {'batch_%s' % name: value for name, value in buddy.last_list_re('^train_')},
                            prefix='train')

            if ii > 0 and ii % 100 == 0: print(('  %d: Average iteration time over last 100 train iters: %.3gs' % (ii, toc3() / 100))); tic3()

            ## DEBUG

            ## dynamic p_size and n_size, shouldn slightly very every sample 
            #if ii > 0 and ii % 100 == 0:
            #    print 'TRAIN --- '
            #    print sess.run(model.p_size, feed_dict=feed_dict)
            #    print sess.run(model.n_size, feed_dict=feed_dict)
            #    valid = sess.run(model.valid_mask, feed_dict=feed_dict)
            #    if not valid.all():
            #        print 'some boxes are invalid'
            #        embed()
            #    #iou = sess.run(model.iou_matrix, feed_dict=feed_dict)
            #    #pos_box_gt_idx, pos_iou = sess.run([model.box_sampler.pos_box_gt_indices, model.pos_iou], feed_dict=feed_dict)
            #    #outbound_mask = sess.run(model.box_sampler.outbound_mask, feed_dict=feed_dict)
            #    #closest_mask = sess.run(model.box_sampler.closest_mask, feed_dict=feed_dict)
            #    #closest_pred_box = sess.run(model.box_sampler.closest_pred_box, feed_dict=feed_dict)
            #    #pos_mask1 = sess.run(model.box_sampler.pos_mask1, feed_dict=feed_dict)
            #    #pos_candi_mask = sess.run(model.box_sampler.pos_candi_mask, feed_dict=feed_dict)
            #    #pos_candi_indx = sess.run(model.box_sampler.pos_candi_indx, feed_dict=feed_dict)
            #    nms_boxes, nms_iou, nms_iou_self, nms_scores, argmax_nms_iou = sess.run([model.nms_boxes, model.nms_iou_matrix, model.nms_iou_self_matrix, model.nms_scores, model.argmax_nms_iou], feed_dict=feed_dict)
            #    embed()



            ## END DEBUG


            buddy.inc_train_iter()   # after finished training a mini-batch

        buddy.inc_epoch()   # after finished training whole pass through set

        if args.output and do_log_train(buddy.epoch, buddy.train_iter, 0):
            log_scalars(writer, buddy.train_iter,
                        {'mean_%s' % name: value for name,value in buddy.epoch_mean_list_re('^train_')},
                        prefix='train')

    print('\nFinal')
    print(('%02d:%d val:   %s' % (buddy.epoch, buddy.train_iter, buddy.epoch_mean_pretty_re('^val_', style=val_style))))
    print(('%02d:%d train: %s' % (buddy.epoch, buddy.train_iter, buddy.epoch_mean_pretty_re('^train_', style=train_style))))

    print('\nEnd of training. Saving evaluation results on whole train and val set.')
    

    if args.output:
        writer.close()   # Flush and close


if __name__ == '__main__':
    main()
