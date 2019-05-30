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


import argparse
from IPython import embed
from matplotlib.pyplot import *

from .loader import load_tvt_single, load_tvt_double, load_tvt_n_per_field, load_tvt_n_per_field_centercrop



def main():
    parser = argparse.ArgumentParser(description='Demo dataset',
                                     formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog)
    )
    parser.add_argument('--num', '-N', type=int, default=2, help='Select the dataset with NUM digits per frame.')
    args = parser.parse_args()

    #train_ims, train_pos, train_class, val_ims, val_pos, val_class, test_ims, test_pos, test_class = load_tvt_single()
    #train_ims, train_pos, train_class, val_ims, val_pos, val_class, test_ims, test_pos, test_class = load_tvt_double()
    #train_ims, train_pos, train_class, val_ims, val_pos, val_class, test_ims, test_pos, test_class = load_tvt_n_per_field(args.num)
    train_ims, train_pos, train_class, val_ims, val_pos, val_class, test_ims, test_pos, test_class = load_tvt_n_per_field_centercrop(args.num)

    print('train_ims   dtype float32, shape (b, 64, 64, 1) = %s' % repr(train_ims.shape))
    print('train_pos   dtype float32, shape (b, N, 4)      = %s' % repr(train_pos.shape))
    print('train_class dtype uint8,   shape (b, N)         = %s' % repr(train_class.shape))

    gray()
    show_ims, show_pos, show_class = train_ims, train_pos, train_class
    #show_ims, show_pos, show_class = val_ims, val_pos, val_class
    #show_ims, show_pos, show_class = test_ims, test_pos, test_class
    img_h, img_w = show_ims.shape[1], show_ims.shape[2]
    for ii in range(16):
        subplot(4,4,ii+1)
        imshow(show_ims[ii].squeeze())
        for jj in range(show_pos[ii].shape[0]):
            i_center, j_center, height, width = show_pos[ii,jj]
            if not(0 < i_center < img_h) or not(0 < j_center < img_w):
                # do not show boxes whose center is outside of canvas
                continue
            gca().add_patch(Rectangle((j_center - width/2, i_center - height/2), width, height, ec='m', fc=(0,0,0,0)))
            text(j_center - width/2, i_center - height/2, '%d' % show_class[ii,jj], color=(1,.5,1))

    print('\nClose plot to exit...')
    show()
    #embed()


if __name__ == "__main__":
    main()
