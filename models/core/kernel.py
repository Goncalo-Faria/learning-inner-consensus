from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import abc

class Kernel(object):
    """
        meta class for kernel functions.  
    """
    __metaclass__ = abc.ABCMeta

    def __init__(
        self,
        name,
        verbose = False):
        self._verbose=verbose
        self.name = name

    @abc.abstractmethod
    def apply(self, a, b):
        ## a,b :: { batch, output_atoms, new_w, new_h, depth * np.prod(ksizes) } + repdim
        raise NotImplementedError('Not implemented')

    def take(self, a, b):
        with tf.name_scope('Kernel/' + self.name) as scope:
            return self.apply(a,b)