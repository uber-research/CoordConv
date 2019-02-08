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

# code to generate dots and squares the red and blue simple shape data used  
import copy
import cv2
import os, sys
import numpy as np
import random
import h5py
from IPython import embed
from collections import defaultdict

MAGIC_SEED_OF_GREATNESS = 10101
MAGIC_SEED_OF_WONDER = 20202

# colors in BGR
colors = {
    'b':(0,0,255),##b
    'g':(0,255,0),##g
    'r':(255,0,0),##r
    'o':(255,156,0),##o
    'gr':(128,128,128),
    'k':(0,0,0),##k
    'y':(255,255,0),##y,
    'w':(255,255,255)##w
}

"""
Class that will place and orient a set of objects
"""
class canvas(object):
    def __init__(self,img_size,overlap_allowed=True,bgcolor=(255,255,255)):
        self.img_size = img_size
        self.bgcolor = bgcolor

        #white background
        self.img = np.ones((img_size,img_size,3)) 

        self.overlap_allowed = overlap_allowed
        self.clear()

    def clear(self):
        """clear canvas"""

        #set image to background color
        self.img[:,:,0] = self.bgcolor[0]
        self.img[:,:,1] = self.bgcolor[1]
        self.img[:,:,2] = self.bgcolor[2]

        #clear objects
        self.objects = []

    def place_object(self,obj,center=None):
        """
        randomly place an image on the canvas (unless location is specified)
        """
        new_obj = copy.deepcopy(obj)

        try:
            if center==None:
                new_obj.center = self.center_generate(new_obj)
        except:
            new_obj.center = center

        self.objects.append(new_obj)

    def center_generate(self,obj):
        """
        randomly generate canvas location (respecting overlap constraint)
        """
        while True:
            pas = True

            center = np.random.randint(0+obj.size, self.img_size - obj.size, 2)
            if self.overlap_allowed:
                return center

            #collision detection
            if len(self.objects) > 0:
                for _obj in self.objects:
                    if ((center - _obj.center) ** 2).sum() < ((obj.size+_obj.size) ** 2):
                        pas = False
            if pas:
                return center

    def onehots(self):
        """
        generate onehot outputs
        """

        #note: objects are sorted by increasing depth -- this will determine
        #ordering of returned one-hot feature maps
        self.objects.sort(key=lambda x:x.depth)
        
        onehots = []
        for obj in self.objects:
            onehots.append(obj.generate_center_onehot())

        stacked_onehots = np.concatenate(onehots,axis=2)
        return stacked_onehots

    def get_locations(self):
        """
        generate xy coordinates
        """

        #note: objects are sorted by increasing depth -- this will determine
        #ordering of returned one-hot feature maps
        self.objects.sort(key=lambda x:x.depth)

        coords = []
        for obj in self.objects:
            coords.append(obj.center)
        coords = np.array(coords).flatten()
        stacked_coords = np.stack(coords)
        return stacked_coords

    def get_normalized_locations(self):
        """
        generate xy coordinates
        """

        norm_locs = self.get_locations()/float(self.img_size)
        return norm_locs

    def render(self):
        #before rendering sort objects by increasing depth
        self.objects.sort(key=lambda x:x.depth)

        for obj in self.objects:
            obj.render(self.img)

        return self.img.copy()


"""
A single instance of a visual object
"""
class visual_object(object):
    def __init__(self,shape,size,color,depth=0,center_x=0,center_y=0):
        self.shape = shape
        self.size = size
        self.center = np.array((center_x,center_y))
        self.color = color
        self.depth = depth

    def generate_center_onehot(self):
        """generate a one-hot image"""

        #note: numpy image array indices follow (y,x) convention 
        #but cv2 coordinates follow (x,y) convention
        field = np.zeros((img_size,img_size,1))
        field[self.center[1],self.center[0],0]=1.0

        return field

    def render(self,img):
        center,size = self.center,self.size
        color = colors[self.color]

        if self.shape=='rectangle':
            start = (center[0]-size, center[1]-size)
            end = (center[0]+size, center[1]+size)
            cv2.rectangle(img, start, end, color, -1)

        if self.shape=='circle':
            center_ = (center[0], center[1])
            cv2.circle(img, center_, size, color, -1)

class multiobject_fixed_generator(object):
    """data generator that uses two fixed objects"""
    def __init__(self,img_size,objects,num_samples,bgcolor=(255,255,255)):
        """
        img_size: size of the canvas
        obj: object to drag over the canvas to generate data
        """
        self.canvas = canvas(img_size,bgcolor=bgcolor,overlap_allowed=False)
        self.data = [] 
        self.objects = objects
        self.num_objects = len(objects)
        self.num_samples = num_samples
        self.img_size = img_size
        self.obj_size = 4
        self.generate()

    def size(self):
        return len(self.data)

    def generate(self,step=1):

        for _idx in range(self.num_samples):

            self.canvas.clear()
            
            for obj in self.objects:
                self.canvas.place_object(obj)

            img = self.canvas.render()
            one_hots = self.canvas.onehots()
            locations = self.canvas.get_locations()
            normalized_locations = self.canvas.get_normalized_locations()

            datapoint = {}
            datapoint['image']=img
            datapoint['imagegray']=img.mean(-1)[:,:,np.newaxis]
            datapoint['onehots']=one_hots
            datapoint['locations']=locations
            datapoint['normalized_locations']=normalized_locations

            self.data.append(datapoint)


class multiobject_random_generator(object):
    """data generator that sweeps an object over all possible locations"""
    def __init__(self,img_size,num_objects,num_samples,bgcolor=(255,255,255)):
        """
        img_size: size of the canvas
        obj: object to drag over the canvas to generate data
        """
        self.canvas = canvas(img_size,bgcolor=bgcolor,overlap_allowed=False)
        self.data = [] 
        self.num_objects = num_objects
        self.num_samples = num_samples
        self.img_size = img_size
        self.obj_size = 4
        self.generate()

    def size(self):
        return len(self.data)

    def generate(self,step=1):

        for _idx in range(self.num_samples):
            self.canvas.clear()
            objs = []
            for _obj in range(self.num_objects):
                color = random.choice(colors.keys()) #beware this rng
                shape = random.choice(['rectangle','circle'])
                size = self.obj_size

                obj = visual_object(shape, size, color, _obj)
                self.canvas.place_object(obj)

            img = self.canvas.render()
            one_hots = self.canvas.onehots()
            locations = self.canvas.get_locations()
            normalized_locations = self.canvas.get_normalized_locations()

            datapoint = {}
            datapoint['image']=img
            datapoint['imagegray']=img.mean(-1)[:,:,np.newaxis]
            datapoint['onehots']=one_hots
            datapoint['locations']=locations
            datapoint['normalized_locations']=normalized_locations

            self.data.append(datapoint)

class objectsweep_generator(object):
    """data generator that sweeps an object over all possible locations"""
    def __init__(self,img_size,obj,bgcolor=(255,255,255)):
        """
        img_size: size of the canvas
        obj: object to drag over the canvas to generate data
        """
        self.canvas = canvas(img_size,bgcolor=bgcolor)
        self.data = [] 
        self.object = obj
        self.img_size = img_size
        self.generate()

    def size(self):
        return len(self.data)

    def generate(self,step=1):
        obj = self.object

        for x in range(obj.size,self.img_size-obj.size,step):
            for y in range(obj.size,self.img_size-obj.size,step):
                self.canvas.clear()
                self.canvas.place_object(obj,center=np.array((x,y)))

                img = self.canvas.render()
                one_hots = self.canvas.onehots()
                locations = self.canvas.get_locations()
                normalized_locations = self.canvas.get_normalized_locations()

                datapoint = {}
                datapoint['image']=img
                datapoint['imagegray']=img.mean(-1)[:,:,np.newaxis]
                datapoint['onehots']=one_hots
                datapoint['locations']=locations
                datapoint['normalized_locations']=normalized_locations

                self.data.append(datapoint)

def quadrant_split(datapoint):
    if datapoint['normalized_locations'][0] > 0.5 and datapoint['normalized_locations'][1] > 0.5:
            return True
    else:
        return False

def random_split(datapoint,pct=0.25):
    if np.random.random()<pct:
        return True
    return False

#stick into val set if *any* object's coordinate is in val zone
def checkarray_split(datapoint,array):
    locs = datapoint['locations']

    for k in range(0,len(locs),2):
        if tuple(locs[k:k+2]) not in array:
            return True

    return False


class dataset_writer(object):
    """
    generate a h5 dataset

    generator: class that generates data
    x_field: what property to use as x
    y_field: what property to use as y
    valsplit_func: function that assigns datapoints to train or val set
    filename: save name

    """
    def __init__(self,generator,valsplit_func,filename):
        self.train = []
        self.val = []
        self.data = generator.data

        for idx in range(generator.size()):
            if valsplit_func(self.data[idx]):
                self.val.append(self.data[idx])
            else:
                self.train.append(self.data[idx])

        self.train_np = self.to_numpy(self.train)
        self.val_np = self.to_numpy(self.val)

        self.to_h5(filename)

    def to_numpy(self,datapoints,shuffle=True):
        fields = defaultdict(list) 

        for d in datapoints:
            for field in d:
                fields[field].append(d[field])

        np_dict = {}

        np.random.seed(seed=MAGIC_SEED_OF_GREATNESS)
        order = np.random.permutation(len(fields[fields.keys()[0]]))

        for field in fields:
            np_dict[field] = np.array(fields[field])

            if shuffle:
             np_dict[field] = np_dict[field][order]

        return np_dict

    def to_h5(self,filename):

        _filename = os.path.join(filename+".h5")

        #for some reason without deleting old file, opening
        #h5 with 'w' flag will lead to a corrupted archive
        if os.path.exists(_filename):
            answer = raw_input('Overwriting existing file {}: [y]/n '.format(_filename))
            if not answer or answer[0].lower() != 'n':
                os.system("rm %s"%_filename)
            else:
                return

        ff = h5py.File(_filename, 'w')
        for field in self.train_np:
            ff.create_dataset('train_%s'%field, data=self.train_np[field])
            ff.create_dataset('val_%s'%field,data = self.val_np[field])

        #ff.create_dataset('train_y', data=self.tr_y)
        #ff.create_dataset('val_x', data=self.val_x)
        #ff.create_dataset('val_y', data=self.val_y)
        ff.close()
        return

if __name__=='__main__':

    test = False
    twoobjects = False

    #random test code
    if test:
        _canvas = canvas(64,overlap_allowed=False)

        _circle = visual_object('circle',10,'r',depth=10)
        _rectangle = visual_object('rectangle',10,'b',depth=20)
        _rectangle2 = visual_object('rectangle',10,'r',depth=30)

        _canvas.place_object(_circle)
        _canvas.place_object(_rectangle)
        _canvas.place_object(_rectangle2)

        locs = _canvas.get_normalized_locations()
        img = _canvas.render()
        onehots = _canvas.onehots()

        from pylab import *
        imshow(onehots)
        figure()
        imshow(img/255.0)
        show()

    #image size
    img_size = 64

    #object size
    obj_size = 4

    #create size 9 objects (radius 4)
    _circle = visual_object('circle',obj_size,'w',depth=10)
    _rectangle = visual_object('rectangle',obj_size,'gr',depth=12)

    # Generate data containing one object
    #title = "onecircle"
    #generator = objectsweep_generator(img_size,_circle,bgcolor=(0,0,0))
    title = "rectangle"
    generator = objectsweep_generator(img_size,_rectangle,bgcolor=(0,0,0))
    for holdout in ["uniform","quadrant"]:
            np.random.seed(MAGIC_SEED_OF_WONDER)
            h5_name = "%s_%d_%s" % (title, obj_size, holdout)
            #h5_name = "%s_%s_%d_%s" % (title,shape,obj_size,holdout)
            if holdout=='uniform':
                split_function = random_split
            else:
                split_function = quadrant_split
            writer = dataset_writer(generator,split_function,h5_name)

    if twoobjects:

        # Generate data containing two objects
        title = "twofixedobjs"
        generator = multiobject_fixed_generator(img_size,[_circle,_rectangle],num_samples=5000,bgcolor=(0,0,0))

        def generate_coordinate_split(_type='uniform'):

            #create train/val split 
            coord_range = range(obj_size,img_size-obj_size)
            #create mesh of all possible x,y object coordinates
            x,y = np.meshgrid(coord_range,coord_range)

            x=x.flatten()
            y=y.flatten()
            order = np.random.permutation(x.shape[0])
            x=x[order]
            y=y[order]

            training_coords = set()
            val_coords = set()
        
            if _type=='uniform':
                val_split_ratio = 0.1
                total_examples = x.shape[0]
                val_idx = int(total_examples*(1.0-val_split_ratio))
                train_x = x[:val_idx]
                train_y = y[:val_idx]
                val_x = x[val_idx:]
                val_y = y[val_idx:]

                for _idx in range(val_x.shape[0]):
                    val_coords.add((val_x[_idx],val_y[_idx]))

                for _idx in range(train_x.shape[0]):
                    training_coords.add((train_x[_idx],train_y[_idx]))

            if _type=='quadrant':
                midpoint = img_size //2

                for _idx in range(x.shape[0]):
                    coord = (x[_idx],y[_idx])
                    if x[_idx] > midpoint and y[_idx] > midpoint:
                        val_coords.add(coord)
                    else:
                        training_coords.add(coord)

            return training_coords,val_coords

        for holdout in ["uniform",'quadrant']:
                np.random.seed(MAGIC_SEED_OF_WONDER)
                #h5_name = "%s_%d_%s" % (shape,obj_size,holdout)
                h5_name = "%s_%d_%s" % (title,obj_size,holdout)

                if holdout=='uniform':
                    training_coords,val_coords = generate_coordinate_split('uniform')
                elif holdout=='quadrant':
                    training_coords,val_coords = generate_coordinate_split('quadrant')

                split_function = lambda x:checkarray_split(x,training_coords)

                writer = dataset_writer(generator,split_function,h5_name)

