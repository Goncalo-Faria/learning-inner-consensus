from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ..core.metric import Metric


class SquaredFrobenius(Metric):

    def __init__(
            self):
        super(SquaredFrobenius, self).__init__(
            "SquaredFrobenius",
            False)

    def apply(self, a):
        ## a :: { batch, output_atoms, new_w, new_h, depth * np.prod(ksizes) } + repdim
        return tf.reduce_sum(tf.pow(a, 2), axis=[-1, -2], keepdims=True)


class Frobenius(SquaredFrobenius):

    def __init__(
            self,
            epsilon = 1e-6):
        self._epsilon = epsilon
        super(Frobenius, self).__init__()

    def apply(self, a):
        ## a :: { batch, output_atoms, new_w, new_h, depth * np.prod(ksizes) } + repdim
        with tf.name_scope("Frobenius/"):
            sq = super(Frobenius, self).apply(a)
            return tf.sqrt( sq + self._epsilon)
