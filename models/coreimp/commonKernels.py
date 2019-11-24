from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from ..core import variables
from ..core.kernel import Kernel


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


class DotProd(Kernel):
    def __init__(
            self):

        super(DotProd, self).__init__(
            "Dotprod",
            False)

    def apply(self, a, b):
        shape = a.shape.as_list()
        a = tf.reshape(a, shape[:-2] + [1, -1])
        b = tf.reshape(b, shape[:-2] + [-1, 1])

        return tf.matmul(a, b)


class GaussianKernel(Kernel):

    def __init__(
            self,
            verbose=False,
            singular=True):

        self._singular = singular

        super(GaussianKernel, self).__init__(
            "GaussianKernel",
            verbose=verbose)

    def apply(self, a, b):
        exponent = tf.reduce_sum( tf.pow(a - b, 2), axis=[-2,-1], keepdims=True)

        if self._singular:
            length_scale = variables.weight_variable(
                [],
                name="length_scale"
            )
        else:
            length_scale = variables.weight_variable(
                [1, exponent.shape[1], 1, 1, 1, 1, 1],
                name="length_scale"
            )

        rbf = exponent*(-1/2)*(1/(length_scale**2))

        return tf.compat.v1.exp(rbf)


class SpectralMixture(Kernel):
    ## https://arxiv.org/pdf/1302.4245.pdf
    ## https://arxiv.org/pdf/1511.02222.pdf
    def __init__(
            self,
            verbose=False):

        super(SpectralMixture, self).__init__(
            "SpectralMixture",
            verbose=verbose)

    def apply(self, a, b):
        shape = a.shape.as_list()

        a = tf.reshape(a, shape[:-2] + [1, -1])
        b = tf.reshape(b, shape[:-2] + [1, -1])

        ro = a - b

        mu = variables.weight_variable(
            [ro.shape.as_list()[-1]],
            name="mu"
        )

        v = variables.weight_variable(
            [ro.shape.as_list()[-1]],
            name="v"
        )

        atom = ro * np.pi

        prod1 = tf.exp((-2) * (atom ** 2) * v)
        prod2 = tf.cos(atom * mu * 2)

        element_sim = prod1 * prod2

        sim = tf.reduce_sum( element_sim, axis=-1, keepdims = True)

        return sim
