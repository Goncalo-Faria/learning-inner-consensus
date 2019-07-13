from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from core.kernel import Kernel

class Poly(Kernel):
    def __init__(
            self,
            degree):
        self._degree = degree

        super(Poly, self).__init__(
            'poly' + str(self._degree),
            False)

    def apply(self, a, b):
        ## a,b :: { batch, output_atoms, new_w, new_h, depth * np.prod(ksizes) } + repdim

        shape = a.shape.as_list()
        a = tf.reshape(a, shape[:-2] + [1, -1])
        b = tf.reshape(b, shape[:-2] + [-1, 1])

        ## a :: { batch, output_atoms, new_w, new_h, depth * np.prod(ksizes), 1, np.prod(repdim)}
        ## b :: { batch, output_atoms, new_w, new_h, depth * np.prod(ksizes), np.prod(repdim), 1}
        r = tf.pow(tf.matmul(a, b) + 1, self._degree)

        ## r :: { batch, output_atoms, new_w, new_h, depth * np.prod(ksizes), 1, 1}
        return r


class DotProd(Poly):
    def __init__(
            self):
        super(DotProd, self).__init__(
            1)

    def apply(self, a, b):
        return super(DotProd, self).apply(a, b) - 1
