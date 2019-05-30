
# Copyright (c) 2018 Uber Technologies, Inc.
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
import errno
import time



class DuckStruct(object):
    '''Use to store anything!'''

    def __init__(self, **kwargs):
        for k,v in list(kwargs.items()):
            setattr(self, k, v)

    def __repr__(self):
        rep = ['%s=%s' % (k, repr(v)) for k,v in list(self.__dict__.items())]
        return 'DuckStruct(%s)' % ', '.join(rep)


# Mod from: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
class DotDict(dict):
    """
    Example:
    mm = DotDict({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v
        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        if not attr in self:
            raise AttributeError(attr)
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(DotDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, attr):
        if not attr in self:
            raise AttributeError(attr)
        self.__delitem__(attr)

    def __delitem__(self, key):
        super(DotDict, self).__delitem__(key)
        del self.__dict__[key]

    def __repr__(self):
        dict_rep = super(DotDict, self).__repr__()
        return 'DotDict(%s)' % dict_rep


# Mod from https://stackoverflow.com/questions/38987/how-to-merge-two-python-dictionaries-in-a-single-expression
def merge_dicts(*dicts):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    
    ret = {}
    for dd in dicts:
        ret.update(dd)
    return ret


class WithTimer(object):
    def __init__(self, title = '', quiet = False):
        self.title = title
        self.quiet = quiet
        
    def elapsed(self):
        return time.time() - self.wall, time.clock() - self.proc

    def enter(self):
        '''Manually trigger enter'''
        self.__enter__()
    
    def __enter__(self):
        self.proc = time.clock()
        self.wall = time.time()
        return self
        
    def __exit__(self, *args):
        if not self.quiet:
            titlestr = (' ' + self.title) if self.title else ''
            print('Elapsed%s: wall: %.06f, sys: %.06f' % ((titlestr,) + self.elapsed()))


class TicToc(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self._proc = time.clock()
        self._wall = time.time()
        
    def elapsed(self):
        return self.wall(), self.proc()

    def wall(self):
        return time.time() - self._wall

    def proc(self):
        return time.clock() - self._proc


globalTicToc = TicToc()
globalTicToc2 = TicToc()
globalTicToc3 = TicToc()

def tic():
    '''Like Matlab tic/toc for wall time and processor time'''
    globalTicToc.reset()

def toc():
    '''Like Matlab tic/toc for wall time'''
    return globalTicToc.wall()

def tocproc():
    '''Like Matlab tic/toc, but for processor time'''
    return globalTicToc.proc()

def tic2():
    globalTicToc2.reset()
def toc2():
    return globalTicToc2.wall()
def tocproc2():
    return globalTicToc2.proc()
def tic3():
    globalTicToc3.reset()
def toc3():
    return globalTicToc3.wall()
def tocproc3():
    return globalTicToc3.proc()


                
def mkdir_p(path):
    # From https://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def assert_subclass_defines(obj, attr):
    if not hasattr(obj, attr):
        err = 'Tried to fetch %s from object %s, but it is not defined. The subclass must specify a value for %s' % (attr, obj, attr)
        raise Exception(err)


def string_or_gitresman_or_none(value):
    '''Custom validator for --output arg: defaults to
    GIT_RESULTS_MANAGER_DIR variable if it exists and no explicit
    directory was provided.

    If "skip" is provided, leaves blank even if GIT_RESULTS_MANAGER_DIR is populated.
    '''
    if value == 'skip':
        value = ''
    elif value == '':
        value = os.environ.get('GIT_RESULTS_MANAGER_DIR', None)
    return value
