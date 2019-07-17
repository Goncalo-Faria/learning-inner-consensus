from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import tensorflow as tf


class Metric(object):
    """
        meta class for kernel functions.  
    """
    __metaclass__ = abc.ABCMeta

    def __init__(
            self,
            name,
            verbose=False):
        self._verbose = verbose
        self.name = name

    @abc.abstractmethod
    def apply(self, a):
        ## a :: { batch, output_atoms, new_w, new_h, depth * np.prod(ksizes) } + repdim
        raise NotImplementedError('Not implemented')

    def take(self, a):
        with tf.compat.v1.variable_scope('Metric/' + self.name, reuse=tf.compat.v1.AUTO_REUSE) as scope:
            return self.apply(a)
