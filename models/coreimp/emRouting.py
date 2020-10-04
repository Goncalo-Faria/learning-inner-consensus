from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from ..core.routing import RoutingProcedure
from ..core.variables import weight_variable, bias_variable


class EMRouting(RoutingProcedure):

    def __init__(
            self,
            metric,
            iterations,
            name="",
            verbose=False):

        super(EMRouting, self).__init__(
            name="EMRouting" + name,
            metric=metric,
            design_iterations=iterations,
            initial_state=None,
            verbose=verbose)

        self._lambda = 0.01

    def _initial_coefficients(self,activations):

        r = tf.ones(shape= activations.shape,
                    dtype=tf.float32,
                    name="compatibility_value")

        return r

    def _compatibility(self, s, c, votes, poses, probabilities, activations, it):
        ## poses :: { batch, output_atoms, new_w, new_h, 1 } + repdim
        ## votes :: { batch, output_atoms, new_w, new_h, depth * np.prod(ksizes) } + repdim
        ## r :: { batch, output_atoms, new_w , new_h, depth * np.prod(ksizes) ,1 ,1}

        sigma_sq = 2 * s + self._epsilon




        expon = - tf.reduce_sum(tf.pow(votes - poses,2)/sigma_sq,  keepdims=True, axis=[-2,-1])

        logfactor = (-0.5)*tf.reduce_sum( tf.math.log( sigma_sq * np.pi ) , keepdims=True, axis=[-2,-1])
        #factor = 1 / tf.sqrt( tf.reduce_prod( sigma_sq * np.pi , keepdims=True, axis=[-2,-1])/2 )

        pj = tf.exp(logfactor + expon)

        rij = activations * pj

        rij_norm = tf.reduce_sum( rij, keepdims=True, axis=1)

        rij = rij / rij_norm + self._epsilon

        ## nao esta a ser corretamente feito o c_norm
        ## recorer ao repositorio ibm/ implementing em

        #self._norm_coe = tf.reduce_sum(c, keepdims=True, axis=2)

        #c_normed = c / (self._norm_coe + self._epsilon)

        return rij, s

    def _activation(self, s, c, votes, poses):
        ## poses :: { batch, output_atoms, new_w, new_h, 1 } + repdim
        ## votes :: { batch, output_atoms, new_w, new_h, depth * np.prod(ksizes) } + repdim
        ## c :: { batch, output_atoms, new_w , new_h, depth * np.prod(ksizes) }

        vshape = votes.shape.as_list()

        betau = weight_variable(
            [1, vshape[1], 1, 1, 1] + [1, 1],
            name="betau",
            verbose=self._verbose)

        betaa = weight_variable(
            [1, vshape[1], 1, 1, 1] + [1, 1],
            name="betaa",
            verbose=self._verbose)

        sigma_sq = tf.reduce_sum(c * tf.pow(votes - poses, 2) / self._norm_coe, axis=4, keepdims=True)

        costh = self._norm_coe * (betau + 2* tf.math.log(sigma_sq + self._epsilon))

        inverse_temperature = (self._lambda *
                               (1 - tf.pow(0.95, tf.cast(self._it + 1, tf.float32))))

        activation = tf.sigmoid(inverse_temperature * (betaa - tf.reduce_sum(costh, keepdims=True, axis=[-2,-1])))

        return activation, sigma_sq
