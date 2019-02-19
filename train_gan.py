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
import time
import gzip
import cPickle as pickle
import numpy as np
import h5py
import pdb
from IPython import embed
import colorama
import setproctitle
import tensorflow as tf
import tensorflow.contrib.eager as tfe
#
lab_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..')
sys.path.insert(1, lab_root)
#
from general.util import tic, toc, tic2, toc2, tic3, toc3, mkdir_p, WithTimer
from general.vision import ImagePreproc
from general.ffdb import db
#from data.mnist.loader import centered_normed_mnist
from brook.tfutil import hist_summaries_traintest, add_grads_and_vars_hist_summaries, image_summaries_traintest, get_collection_intersection, get_collection_intersection_summary, log_scalars, sess_run_dict, summarize_weights, summarize_opt, tf_assert_all_init, tf_get_uninitialized_variables, add_grad_summaries
from brook.stats_buddy import StatsBuddy
from tf_plus import setup_session_and_seeds, learning_phase, print_trainable_warnings
from model_builders import CoordConvGAN
from exp.tf_nets.util import make_standard_parser
from exp.cgan.utils import save_images, save_average_image, load_sort_of_clevr

arch_choices = ['simple_coordconv_in_g', 'simple_coordconv_in_gd',
                'clevr_coordconv_in_g', 'clevr_coordconv_in_gd', 
                'simple_gan', 'clevr_gan']

def main():
    #tfe.enable_eager_execution()
    parser = make_standard_parser('Train a GAN model on simple one-object binary images or Clevr two-object color images',
                                  arch_choices=arch_choices, skip_train=True, skip_val=True)
    parser.add_argument('--z_dim', type=int, default=10, help='Dimension of noise vector')
    parser.add_argument('--lr2', type=float, default=None, help='learning rate for generator')
    parser.add_argument('--eager', action='store_true', help='enable tf eager mode (experimental).')
    parser.add_argument('--feature_match', '-fm', action='store_true', help='use feature matching loss for generator.')
    parser.add_argument('--feature_match_loss_weight', '-fmalpha', type=float, default=1.0, help='weight on the feature matching loss for generator.')
    parser.add_argument('--pairedz', action='store_true', help='If True, pair the same z with a training batch each epoch')
    parser.add_argument('--eval-train-every', type=int, default=0, help='evaluate whole training set every N epochs. 0 to disable.')
    #parser.add_argument('--clevr', action='store_true', help='load 2-objects clevr data instead of simple data')

    args = parser.parse_args()

    args.skipval = True

    if args.eager:
        tfe.enable_eager_execution()

    minibatch_size = args.minibatch
    train_style, val_style = ('', '') if args.nocolor else (colorama.Fore.BLUE, colorama.Fore.MAGENTA)
    evaltrain_style = '' if args.nocolor or args.eval_train_every <= 0 else colorama.Fore.CYAN
    
    black_divider = True if args.arch.startswith('clevr') else False

    # Get a TF session and set numpy and TF seeds
    if not args.eager:
        sess = setup_session_and_seeds(args.seed, assert_gpu=not args.cpu)

    # 0. LOAD DATA
    if args.arch.startswith('simple'):
        fd = h5py.File('data/rectangle_4_uniform.h5','r')
        train_x = np.array(fd['train_imagegray'], dtype=float)/255.0 # shape (2368, 64, 64, 1)
        val_x = np.array(fd['val_imagegray'], dtype=float)/255.0 # shape (768, 64, 64, 1)
        train_x = np.concatenate((train_x, val_x), axis=0) # shape (3136, 64, 64, 1)

    elif args.arch.startswith('clevr'):
        (train_x, val_x) = load_sort_of_clevr(twoobjs=True)
        train_x = np.concatenate((train_x, val_x), axis=0) # shape (50000, 64, 64, 3)

    else:
        raise Exception('Unknown network architecture: %s' % args.arch)

    print 'Train data loaded: {} images, size {}'.format(train_x.shape[0], train_x.shape[1:])
    #print 'Val data loaded: {} images, size {}'.format(val_x.shape[0], val_x.shape[1:]) 

    #print 'Label dimension: {}'.format(val_y.shape[1:]) 

    # 1. CREATE MODEL
    assert len(train_x.shape)==4, "image data must be of 4 dimensions" 
    image_h, image_w, image_c = train_x.shape[1], train_x.shape[2], train_x.shape[3]

    model = build_model(args, image_h, image_w, image_c)

    print 'All model weights:'
    summarize_weights(model.trainable_weights)
    print 'Model summary:'
    #model.summary()      # TOREPLACE
    print 'Another model summary:'
    model.summarize_named(prefix='  ')
    print_trainable_warnings(model)



    # 2. COMPUTE GRADS AND CREATE OPTIMIZER
    lr_gen = args.lr2 if args.lr2 else args.lr
    
    if args.opt == 'sgd':
        d_opt = tf.train.MomentumOptimizer(args.lr, args.mom)
        g_opt = tf.train.MomentumOptimizer(lr_gen, args.mom)
    elif args.opt == 'rmsprop':
        d_opt = tf.train.RMSPropOptimizer(args.lr, momentum=args.mom)
        g_opt = tf.train.RMSPropOptimizer(lr_gen, momentum=args.mom)
    elif args.opt == 'adam':
        d_opt = tf.train.AdamOptimizer(args.lr, args.beta1, args.beta2)
        g_opt = tf.train.AdamOptimizer(lr_gen, args.beta1, args.beta2)

    # Optimize w.r.t all trainable params in the model

    all_vars = model.trainable_variables
    d_vars = [var for var in all_vars if 'discriminator' in var.name]
    g_vars = [var for var in all_vars if 'generator' in var.name]

    d_grads_and_vars = d_opt.compute_gradients(model.d_loss, d_vars, gate_gradients=tf.train.Optimizer.GATE_GRAPH)
    d_train_step = d_opt.apply_gradients(d_grads_and_vars)
    g_grads_and_vars = g_opt.compute_gradients(model.g_loss, g_vars, gate_gradients=tf.train.Optimizer.GATE_GRAPH)
    g_train_step = g_opt.apply_gradients(g_grads_and_vars)
    
    #with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    #    d_train_step = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(model.d_loss, var_list=d_vars)
    #    g_train_step = tf.train.AdamOptimizer(args.lr2, beta1=args.beta1).minimize(model.g_loss, var_list=g_vars)


    #add_grad_summaries(d_grads_and_vars)
    #add_grad_summaries(g_grads_and_vars)
    #summarize_opt(d_opt)
    #summarize_opt(g_opt)

    #d_grads = [ii[0] for ii in d_grads_and_vars]
    #g_grads = [ii[0] for ii in g_grads_and_vars]

    #d_grads_mean_sq = tf.add_n([tf.reduce_mean(tf.square(ii)) for ii in d_grads]) / len(d_grads)
    #g_grads_mean_sq = tf.add_n([tf.reduce_mean(tf.square(ii)) for ii in g_grads]) / len(g_grads)


    hist_summaries_traintest(model.d_real_logits, model.d_fake_logits)
    
    add_grads_and_vars_hist_summaries(d_grads_and_vars)
    add_grads_and_vars_hist_summaries(g_grads_and_vars)
    image_summaries_traintest(model.fake_images)


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
        buddy = StatsBuddy(pretty_replaces=[('evaltrain_', ''), ('eval','')]) if args.eval_train_every > 0 else StatsBuddy()
        
    buddy.tic()    # call if new run OR resumed run

    # Initialize any missed vars (e.g. optimization momentum, ... if not loaded from checkpoint)
    #uninitialized_vars = tf_get_uninitialized_variables(sess)
    #init_missed_vars = tf.variables_initializer(uninitialized_vars, 'init_missed_vars')
    #sess.run(init_missed_vars)
    ## Print warnings about any TF vs. Keras shape mismatches
    ##warn_misaligned_shapes(model)
    ## Make sure all variables, which are model variables, have been initialized (e.g. model params and model running BN means)
    #tf_assert_all_init(sess)
    tf.global_variables_initializer().run()


    # 4. SETUP TENSORBOARD LOGGING
    #d_logits_real_sum = tf.summary.histogram("d_logits_real", model.d_real_logits)
    #d_logits_fake_sum = tf.summary.histogram("d_logits_fake", model.d_fake_logits)

    #d_grads_mean_sq_sum = tf.summary.scalar("d_grads_mean_sq", d_grads_mean_sq)
    #g_grads_mean_sq_sum = tf.summary.scalar("g_grads_mean_sq", g_grads_mean_sq)

    # final summary operations
    #d_logits_sum = tf.summary.merge([d_logits_real_sum, d_logits_fake_sum])
    #grads_mean_sum = tf.summary.merge([d_grads_mean_sq_sum, g_grads_mean_sq_sum])

    param_histogram_summaries = get_collection_intersection_summary('param_collection', 'orig_histogram')
    train_histogram_summaries = get_collection_intersection_summary('train_collection', 'orig_histogram')
    train_scalar_summaries    = get_collection_intersection_summary('train_collection', 'orig_scalar')
    test_histogram_summaries   = get_collection_intersection_summary('test_collection', 'orig_histogram')
    test_scalar_summaries      = get_collection_intersection_summary('test_collection', 'orig_scalar')
    train_image_summaries     = get_collection_intersection_summary('train_collection', 'orig_image')
    test_image_summaries     = get_collection_intersection_summary('test_collection', 'orig_image')

    
    writer = None
    if args.output:
        mkdir_p(args.output)
        writer = tf.summary.FileWriter(args.output, sess.graph)

    # 5. TRAIN
    train_iters = (train_x.shape[0]) // minibatch_size
    if not args.skipval:
        val_iters = (val_x.shape[0]) // minibatch_size

    if args.ipy:
        print 'Embed: before train / val loop (Ctrl-D to continue)'
        embed()

    # 2. use same noise, eval on 100 samples and save G(z),  
    np.random.seed()
    eval_batch_size = 100
    eval_z = np.random.uniform(-1, 1, size=(eval_batch_size, args.z_dim))
    
    while buddy.epoch < args.epochs + 1:
        # How often to log data
        do_log_params = lambda ep, it, ii: True
        do_log_val = lambda ep, it, ii: True
        do_log_train = lambda ep, it, ii: (it < train_iters and it & it-1 == 0 or it>=train_iters and it%train_iters == 0)  # Log on powers of two then every epoch

        # 0. Log params
        if args.output and do_log_params(buddy.epoch, buddy.train_iter, 0) and param_histogram_summaries is not None:
            params_summary_str, = sess.run([param_histogram_summaries])
            writer.add_summary(params_summary_str, buddy.train_iter)

        # 1. Evaluate generator by showing random generated results 
        #    Evaluate descriminator by showing seeing correct rate on generated and real (hold-out) results
        #assert(args.skipval), "only support training now"

        if not args.skipval:
            tic2()
            # use different noise, eval on larger number of samples and get correct rate       
            np.random.seed()
            val_z = np.random.uniform(-1, 1, size=(val_x.shape[0], args.z_dim))
                
            with WithTimer('sess.run val iter', quiet=not args.verbose):
                feed_dict = {
                    model.input_images: val_x,
                    model.input_noise: val_z,
                    learning_phase(): 0
                    }

                if 'input_labels' in model.named_keys():
                    feed_dict.update({model.input_labels: val_y})

                val_corr_fake_bn0, val_corr_real_bn0 = sess.run([model.correct_fake, model.correct_real],
                                                        feed_dict = feed_dict)
                
                feed_dict[learning_phase()] = 1
                val_corr_fake_bn1, val_corr_real_bn1 = sess.run([model.correct_fake, model.correct_real],
                                                        feed_dict = feed_dict)

            if args.output and do_log_val(buddy.epoch, buddy.train_iter, 0): 
                fetch_dict = {}
                if test_image_summaries is not None:
                    fetch_dict.update({'test_image_summaries':test_image_summaries})
                if test_scalar_summaries is not None:
                    fetch_dict.update({'test_scalar_summaries':test_scalar_summaries})
                if test_histogram_summaries is not None:
                    fetch_dict.update({'test_histogram_summaries':test_histogram_summaries})
                if fetch_dict:
                    summary_strs = sess_run_dict(sess, fetch_dict, feed_dict=feed_dict)


            buddy.note_list(['correct_real_bn0', 'correct_fake_bn0', 'correct_real_bn1', 'correct_fake_bn1'],
                            [val_corr_real_bn0, val_corr_fake_bn0, val_corr_real_bn1, val_corr_fake_bn1],
                            prefix='val_')

            print ('%3d (ep %d) val: %s (%.3gs/ep)' % (buddy.train_iter, buddy.epoch, buddy.epoch_mean_pretty_re('^val_', style=val_style), toc2()))
                
            if args.output and do_log_val(buddy.epoch, buddy.train_iter, 0):
                log_scalars(writer, buddy.train_iter,
                            {'mean_%s' % name: value for name, value in buddy.epoch_mean_list_re('^val_')},
                            prefix='buddy')
                
                if test_image_summaries is not None:
                    image_summary_str = summary_strs['test_image_summaries']
                    writer.add_summary(image_summary_str, buddy.train_iter)
                if test_scalar_summaries is not None:
                    scalar_summary_str = summary_strs['test_scalar_summaries']
                    writer.add_summary(scalar_summary_str, buddy.train_iter)
                if test_histogram_summaries is not None:
                    hist_summary_str = summary_strs['test_histogram_summaries']
                    writer.add_summary(hist_summary_str, buddy.train_iter)

        # In addition, evalutate 1000 more images
        np.random.seed()
        eval_more = np.random.uniform(-1, 1, size=(1000, args.z_dim))
        feed_dict2 = {
            model.input_noise: eval_z, # (100,-) generated outside of loop to keep the same every round
            learning_phase(): 0
            }

        eval_samples_bn0 = sess.run(model.fake_images,
                                  feed_dict = feed_dict2)

        feed_dict2[learning_phase()] = 1
        eval_samples_bn1 = sess.run(model.fake_images,
                                    feed_dict = feed_dict2)
        
        # feed in 10 times because coordconv cannot handle too big of a batch
        for cc in range(10):
            eval_z2 = eval_more[cc*100:(cc+1)*100, :]
            _eval_more_samples = sess.run(model.fake_images, feed_dict = 
                                    {model.input_noise: eval_z2, #(1000,-)
                                    learning_phase(): 0})
            eval_more_samples = _eval_more_samples if cc == 0 else np.concatenate((eval_more_samples, _eval_more_samples), axis=0)

        if args.output:
            mkdir_p('{}/fake_images'.format(args.output))
            # eval_samples_bn*: e.g. (100, 64, 64, 3)
            save_images(eval_samples_bn0, [10, 10], 
                        '{}/fake_images/g_out_bn0_epoch_{}_iter_{}.png'.format(args.output, buddy.epoch, buddy.train_iter),
                        black_divider=black_divider)
            save_images(eval_samples_bn1, [10, 10], 
                        '{}/fake_images/g_out_bn1_epoch_{}.png'.format(args.output, buddy.epoch),
                        black_divider=black_divider)
            save_average_image(eval_more_samples,
                        '{}/fake_images/g_out_averaged_epoch_{}_iter_{}.png'.format(args.output, buddy.epoch, buddy.train_iter))

        # 2. Possiby Snapshot, possibly quit
        if args.output and args.snapshot_to and args.snapshot_every:
            snap_intermed = args.snapshot_every > 0 and buddy.train_iter % args.snapshot_every == 0
            snap_end = buddy.epoch == args.epochs
            if snap_intermed or snap_end:
                # Snapshot
                save_path = saver.save(sess, '%s/%s_%04d.ckpt' % (args.output, args.snapshot_to, buddy.epoch))
                print 'snappshotted model to', save_path
                with gzip.open('%s/%s_misc_%04d.pkl.gz' % (args.output, args.snapshot_to, buddy.epoch), 'w') as ff:
                    saved = {'buddy': buddy}
                    pickle.dump(saved, ff)
                # snapshot sampled images too
                ff = h5py.File('%s/sampled_images_%04d.h5'%(args.output, buddy.epoch), 'w')
                ff.create_dataset('eval_samples_bn0', data=eval_samples_bn0)
                ff.create_dataset('eval_samples_bn1', data=eval_samples_bn1)
                ff.create_dataset('eval_z', data=eval_z)
                ff.create_dataset('eval_z_more', data=eval_more)
                ff.create_dataset('eval_more_samples', data=eval_more_samples)
                ff.close()



        
        # 2. Possiby evaluate the training set
        if args.eval_train_every > 0:
            if buddy.epoch % args.eval_train_every == 0:
                tic2()
                for ii in xrange(train_iters):
                    start_idx = ii * minibatch_size
                    if args.pairedz:
                        np.random.seed(args.seed+ii)
                    else:
                        np.random.seed()
                    batch_z = np.random.uniform(-1, 1, size=(minibatch_size, args.z_dim))

                    batch_x = train_x[start_idx:start_idx + minibatch_size]
                    batch_y = train_y[start_idx:start_idx + minibatch_size]               
                            
                    feed_dict = {
                        model.input_images: batch_x,
                        #model.input_labels: batch_y,
                        model.input_noise: batch_z,
                        learning_phase(): 0,
                    }
                    
                    if 'input_labels' in model.named_keys():
                        feed_dict.update({model.input_labels: val_y})

                    fetch_dict = model.trackable_dict()
                    result_eval_train = sess_run_dict(sess, fetch_dict, feed_dict=feed_dict)
                    buddy.note_weighted_list(batch_x.shape[0], model.trackable_names(), [result_eval_train[k] for k in model.trackable_names()], prefix='evaltrain_bn0_')

                    feed_dict = {
                        model.input_images: batch_x,
                        #model.input_labels: batch_y,
                        model.input_noise: batch_z,
                        learning_phase(): 1,
                    }
                    if 'input_labels' in model.named_keys():
                        feed_dict.update({model.input_labels: val_y})

                    result_eval_train = sess_run_dict(sess, fetch_dict, feed_dict=feed_dict)
                    buddy.note_weighted_list(batch_x.shape[0], model.trackable_names(), [result_eval_train[k] for k in model.trackable_names()], prefix='evaltrain_bn1_')

                    if args.output:
                        log_scalars(writer, buddy.train_iter,
                                    {'batch_%s' % name: value for name, value in buddy.last_list_re('^evaltrain_bn0_')}, prefix='buddy')
                        log_scalars(writer, buddy.train_iter,
                                    {'batch_%s' % name: value for name, value in buddy.last_list_re('^evaltrain_bn1_')}, prefix='buddy')
                if args.output:
                    log_scalars(writer, buddy.epoch,
                                {'mean_%s' % name: value for name,value in buddy.epoch_mean_list_re('^evaltrain_bn0_')}, prefix='buddy')
                    log_scalars(writer, buddy.epoch,
                                {'mean_%s' % name: value for name,value in buddy.epoch_mean_list_re('^evaltrain_bn1_')}, prefix='buddy')

                print ('%3d (ep %d) evaltrain: %s (%.3gs/ep)' % (buddy.train_iter, buddy.epoch, buddy.epoch_mean_pretty_re('^evaltrain_bn0_', style=evaltrain_style), toc2()))
                print ('%3d (ep %d) evaltrain: %s (%.3gs/ep)' % (buddy.train_iter, buddy.epoch, buddy.epoch_mean_pretty_re('^evaltrain_bn1_', style=evaltrain_style), toc2()))

        
        if buddy.epoch == args.epochs:
            if args.ipy:
                print 'Embed: at end of training (Ctrl-D to exit)'
                embed()
            break   # Extra pass at end: just report val stats and skip training

        # 3. Train on training set

        if args.shuffletrain:
            train_order = np.random.permutation(train_x.shape[0])
            train_order2 = np.random.permutation(train_x.shape[0])
        tic3()
        for ii in xrange(train_iters):
            tic2()
            start_idx = ii * minibatch_size
            if args.pairedz:
                np.random.seed(args.seed+ii)
            else:
                np.random.seed()

            batch_z = np.random.uniform(-1, 1, size=(minibatch_size, args.z_dim))

            if args.shuffletrain:
                #batch_x = train_x[train_order[start_idx:start_idx + minibatch_size]]
                batch_x = train_x[sorted(train_order[start_idx:start_idx + minibatch_size].tolist())]
                if args.feature_match:
                    assert args.shuffletrain, "feature matching loss requires shuffle train"
                    batch_x2 = train_x[sorted(train_order2[start_idx:start_idx + minibatch_size].tolist())]
                if 'input_labels' in model.named_keys():
                    batch_y = train_y[sorted(train_order[start_idx:start_idx + minibatch_size].tolist())]
            else:
                batch_x = train_x[start_idx:start_idx + minibatch_size]
                if 'input_labels' in model.named_keys():
                    batch_y = train_y[start_idx:start_idx + minibatch_size]               
                            
            feed_dict = {
                model.input_images: batch_x,
                #model.input_labels: batch_y,
                model.input_noise: batch_z,
                learning_phase(): 1,
            }
            
            if 'input_labels' in model.named_keys():
                feed_dict.update({model.input_labels: batch_y})
            if 'input_images2' in model.named_keys():
                feed_dict.update({model.input_images2: batch_x2})

            fetch_dict = model.trackable_and_update_dict()
            
            if args.output and do_log_train(buddy.epoch, buddy.train_iter, ii):
                if train_histogram_summaries is not None:
                    fetch_dict.update({'train_histogram_summaries': train_histogram_summaries})
                if train_scalar_summaries is not None:
                    fetch_dict.update({'train_scalar_summaries': train_scalar_summaries})
                if train_image_summaries is not None:
                    fetch_dict.update({'train_image_summaries': train_image_summaries})


            with WithTimer('sess.run train iter', quiet=not args.verbose):
                result_train = sess_run_dict(sess, fetch_dict, feed_dict=feed_dict)
                
                #if result_train['d_loss'] < result_train['g_loss']:
                #    #print 'Only train G'
                #    sess.run(g_train_step, feed_dict=feed_dict)
                #else:
                #    #print 'Train both D and G'
                #    sess.run(d_train_step, feed_dict=feed_dict)
                #    sess.run(g_train_step, feed_dict=feed_dict)
                #    sess.run(g_train_step, feed_dict=feed_dict)
                sess.run(d_train_step, feed_dict=feed_dict)
                sess.run(g_train_step, feed_dict=feed_dict)
                sess.run(g_train_step, feed_dict=feed_dict)


            if do_log_train(buddy.epoch, buddy.train_iter, ii):
                buddy.note_weighted_list(batch_x.shape[0], model.trackable_names(), [result_train[k] for k in model.trackable_names()], prefix='train_')
                print ('[%5d] [%2d/%2d] train: %s (%.3gs/i)' % (buddy.train_iter, buddy.epoch, args.epochs, buddy.epoch_mean_pretty_re('^train_', style=train_style), toc2()))

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
                            prefix='buddy')

            if ii > 0 and ii % 100 == 0: print '  %d: Average iteration time over last 100 train iters: %.3gs' % (ii, toc3() / 100); tic3()

            buddy.inc_train_iter()   # after finished training a mini-batch

        buddy.inc_epoch()   # after finished training whole pass through set

        if args.output and do_log_train(buddy.epoch, buddy.train_iter, 0):
            log_scalars(writer, buddy.train_iter,
                        {'mean_%s' % name: value for name,value in buddy.epoch_mean_list_re('^train_')},
                        prefix='buddy')

    print '\nFinal'
    print '%02d:%d val:   %s' % (buddy.epoch, buddy.train_iter, buddy.epoch_mean_pretty_re('^val_', style=val_style))
    print '%02d:%d train: %s' % (buddy.epoch, buddy.train_iter, buddy.epoch_mean_pretty_re('^train_', style=train_style))

    print '\nfinal_stats epochs %g' % buddy.epoch
    print 'final_stats iters %g' % buddy.train_iter
    print 'final_stats time %g' % buddy.toc()
    for name, value in buddy.epoch_mean_list_all():
        print 'final_stats %s %g' % (name, value)

    if args.output:
        writer.close()   # Flush and close

def build_model(args, image_h, image_w, image_c):
    with WithTimer('Make model'):
        input_images = tf.placeholder(shape=(None, image_h, image_w, image_c), dtype='float32')
        input_labels = tf.placeholder(shape=(None,), dtype='int32')
        input_noise = tf.placeholder(shape=(None, args.z_dim), dtype='float32')
        if args.feature_match:
            input_images2 = tf.placeholder(shape=(None, image_h, image_w, image_c), dtype='float32')

        if args.arch.endswith('gan'):
            model = CoordConvGAN(l2=args.l2, x_dim=image_h, y_dim=image_w, cout=image_c, 
                            coords_in_g=False, coords_in_d=False)

        elif args.arch.endswith('coordconv_in_g'):
            model = CoordConvGAN(l2=args.l2, x_dim=image_h, y_dim=image_w, cout=image_c, 
                            coords_in_g=True, coords_in_d=False)
        
        elif args.arch.endswith('coordconv_in_gd'):
            model = CoordConvGAN(l2=args.l2, x_dim=image_h, y_dim=image_w, cout=image_c,
                            coords_in_g=True, coords_in_d=True)
    
        else:
            raise Exception('Unknown network architecture: %s' % args.arch)

        model.a('input_images', input_images)
        model.a('input_noise', input_noise)

        if args.feature_match:
            model.a('input_images2', input_images2)

        # call model on inputs
        input_list = [input_images, input_noise, input_images2] if args.feature_match else [input_images, input_noise]
        model(input_list, feature_matching_loss=args.feature_match,
              feature_match_loss_weight=args.feature_match_loss_weight)
        return model

if __name__ == '__main__':
    main()
