from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import abc

class Transform(object):
    """
        meta class for viewpoint invarient transformations.  
    """
    __metaclass__ = abc.ABCMeta

    def __init__(
        self,
        name,
        output_atoms,
        verbose = False):
        self._verbose=verbose
        self._output_atoms = output_atoms
        self.name = name


    def translate(self, poses, activations):
        with tf.name_scope('Transform/' + self.name) as scope:
            return self.apply(poses, activations)

    @abc.abstractmethod
    def apply(self, poses, activations):
        ## poses { batch, new_w , new_h, depth * np.prod(ksizes) } + repdim
        ## W :: {1, 1 , 1, depth * np.prod(ksizes) } + repdim
        raise NotImplementedError('Not implemented')