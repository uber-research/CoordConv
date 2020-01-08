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

# code to generate sort-of-clevr dataset  
# modified from: https://github.com/kimhc6028/relational-networks.git
# limited to two colors, two shapes

import cv2
import os, sys
import numpy as np
import random
import pickle
import h5py
#import scipy.misc  # Deprecated
import imageio 
from IPython import embed

lab_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
sys.path.insert(1, lab_root)
#
from general.util import mkdir_p

train_size = 50000
test_size = 200
img_size = 64
size = 10 

#dirs = os.path.join("./", "sort_of_clevr")
dirs = os.path.join(os.path.abspath(os.path.dirname(__file__)), "sort_of_clevr")

mkdir_p(dirs)

# colors in rgb
colors = [
    (0,0,255),##b
#    (0,255,0),##g
    (255,0,0),##r
#    (255,156,0),##o
#    (128,128,128),##k
#    (255,255,0)##y
]


def center_generate(objects):
    while True:
        pas = True
        center = np.random.randint(0+size, img_size - size, 2)        
        if len(objects) > 0:
            for name,c,shape in objects:
                if ((center - c) ** 2).sum() < ((size * 2) ** 2):
                    pas = False
        if pas:
            return center



def build_dataset_img_only():
    objects = []
    img = np.ones((img_size,img_size,3)) * 255
    for color_id,color in enumerate(colors):  
        center = center_generate(objects)
        if random.random()<0.5:
            start = (center[0]-size, center[1]-size)
            end = (center[0]+size, center[1]+size)
            cv2.rectangle(img, start, end, color, -1)
            objects.append((color_id,center,'r'))
        else:
            center_ = (center[0], center[1])
            cv2.circle(img, center_, size, color, -1)
            objects.append((color_id,center,'c'))
    
    img = img/127.5 - 1.
    return (img, objects)

print('building train datasets...')
train_datasets = [build_dataset_img_only() for _ in range(train_size)]



#print('saving datasets...')
#filename = os.path.join(dirs,'sort-of-clevr.pickle')
#with  open(filename, 'wb') as f:
#    pickle.dump((train_datasets, test_datasets), f)
#print('datasets saved at {}'.format(filename))


train_imgs = []
train_msgs = []
for dtuple in train_datasets:
    train_imgs.append(dtuple[0])
    train_msgs.append(dtuple[1])
train_x = np.array(train_imgs)

#test_imgs = []
#for dtuple in test_datasets:
#    test_imgs.append(dtuple[0])
#test_x = np.array(test_imgs)

print('saving first 10 images...')
for img_count in range(10):
    #path = os.path.join(dirs,'img{}_{}x{}bgr.png'.format(img_count, img_size, img_size))
    path1 = os.path.join(dirs,'img{}_{}x{}rgb.png'.format(img_count, img_size, img_size))
    image = (train_x[img_count]+1)*127.5
    #scipy.misc.imsave(path1, image) # Deprecated
    imageio.imwrite(path1, image)
    
    #path = os.path.join(dirs,'img{}_512x512bgr.png'.format(img_count))
    path1 = os.path.join(dirs,'img{}_512x512rgb.png'.format(img_count))
    image = cv2.resize((train_x[img_count]+1)*127.5, (512,512))
    #scipy.misc.imsave(path1, image) # Deprecated
    imageio.imwrite(path1, image)

filename = 'sort_of_clevr_{}objs_{}rad_{}imgs_{}x'.format(len(colors), size, train_size, img_size)

print('saving image data into h5 ...')
ff = h5py.File(os.path.join(dirs, filename+'.h5'), 'w')
ff.create_dataset('train_x', data=train_x)
#ff.create_dataset('test_x', data=test_x)
ff.close()

print('saving meta data as pickle ...')
filename = os.path.join(dirs,filename+'.meta.pickle')
with  open(filename, 'wb') as f:
    pickle.dump(train_msgs, f)

print('both image data and meta data saved')
