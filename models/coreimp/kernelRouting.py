from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ..core.routing import RoutingProcedure
from ..core.variables import weight_variable, bias_variable


class KernelRouting(RoutingProcedure):

    def __init__(
            self,
            kernel,
            metric,
            iterations,
            name="",
            verbose=False):
        self._kernel = kernel

        super(KernelRouting, self).__init__(
            name="KernelRouting" + name,
            metric=metric,
            design_iterations=iterations,
            initial_state=None,
            verbose=verbose)

    def _compatibility(self, s, r, votes, poses, probabilities, it):
        ## poses :: { batch, output_atoms, new_w, new_h, 1 } + repdim
        ## votes :: { batch, output_atoms, new_w, new_h, depth * np.prod(ksizes) } + repdim
        ## r :: { batch, output_atoms, new_w , new_h, depth * np.prod(ksizes) }

        poses_tiled = tf.tile(poses, [1, 1, 1, 1, self.atoms, 1, 1])
        ## r :: { batch, output_atoms, new_w , new_h, depth * np.prod(ksizes) }

        alpha = weight_variable([],
                                name="alpha",
                                verbose = True,
                                initializer=tf.compat.v1.keras.initializers.constant(value=1.0))

        r = alpha * self._kernel.take(poses_tiled, votes)
        return r, s

    def _activation(self, s, c, votes, poses):
        ## poses :: { batch, output_atoms, new_w, new_h, 1 } + repdim
        ## votes :: { batch, output_atoms, new_w, new_h, depth * np.prod(ksizes) } + repdim
        ## c :: { batch, output_atoms, new_w , new_h, depth * np.prod(ksizes) }

        distance = self.metric.take(votes - poses)

        if self._verbose:
            tf.compat.v1.summary.histogram("distance_in_it_" + str(self._it), distance)

        raw = tf.reduce_sum(tf.multiply(c, distance), axis=-3, keepdims=True)

        ## raw :: { batch, output_atoms, new_w, new_h, 1 } 

        theta1 = weight_variable([1], name="theta1")
        theta2 = bias_variable([1], name="theta2")

        activation = tf.sigmoid(theta1 * raw + theta2)
        ## activation :: { batch, output_atoms, new_w, new_h, 1 } 

        return activation

    def _initial_coefficients(self, r, activations):
        return tf.ones(activations.shape.as_list() + [1, 1], dtype=tf.float32) / activations.shape.as_list()[-1]


class KernelRoutingWithPrior(KernelRouting):

    def __init__(
            self,
            kernel,
            name,
            metric,
            iterations,
            verbose=False):
        super(KernelRoutingWithPrior, self).__init__(
            kernel=kernel,
            name="withPrior/"+ name,
            metric=metric,
            iterations=iterations,
            verbose = verbose)

    def _initial_coefficients(self, r, activations):
        return tf.reshape(activations, shape=activations.shape.as_list() + [1, 1])
