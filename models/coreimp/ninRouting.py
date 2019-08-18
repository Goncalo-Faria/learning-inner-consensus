from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from models.core.routing import HyperSimplifiedRoutingProcedure
import models.core.variables as variables
from models.coreimp.commonKernels import GaussianKernel


class NiNRouting(HyperSimplifiedRoutingProcedure):

    def __init__(
            self,
            metric,
            activation_layers=[],
            compatibility_layers=[],
            bias=False,
            name="",
            epsilon=1e-6,
            normalization=tf.nn.softmax,
            verbose=False,
            rate=0.5,
            train=False):
        self._activation_layers = activation_layers
        self._compatibility_layers = compatibility_layers
        self._rate = rate
        self._train = train

        super(NiNRouting, self).__init__(
            name="NiNRouting_" + name,
            metric=metric,
            verbose=verbose,
            bias=bias,
            epsilon=epsilon,
            normalization=normalization)

    def _compatibility(self, s, r, votes, poses, probabilities, activations, it):
        ## poses :: { batch, output_atoms, new_w, new_h, 1 } + repdim
        ## votes :: { batch, output_atoms, new_w, new_h, depth * np.prod(ksizes) } + repdim
        ## activations :: { batch, output_atoms, new_w, new_h, depth * np.prod(ksizes) } + [1,1]
        ## r :: { batch, output_atoms, new_w , new_h, depth * np.prod(ksizes) }

        vshape = votes.shape.as_list()
        # s :: {degree}

        votes_flatten = tf.reshape(votes, shape=vshape[:-2] + [-1])

        activations_flatten = tf.reshape(activations, shape=vshape[:-2] + [-1])

        capsule_flatten = tf.concat([votes_flatten, activations_flatten], axis=-1)

        local_capsules_flatten = tf.reshape(capsule_flatten, shape=vshape[:-3] + [-1])

        batched_features = tf.reshape(local_capsules_flatten, [-1, local_capsules_flatten.shape.as_list()[-1]])

        counter = 0

        for layer_num in self._compatibility_layers:
            batched_features = tf.compat.v1.layers.Dropout(rate=self._rate)(
                    tf.compat.v1.layers.Dense(
                        units=layer_num,
                        activation=tf.nn.relu,
                        _reuse=tf.compat.v1.AUTO_REUSE,
                        name="l_" + str(counter)
                    )(batched_features),
                    training=self._train
                )

            counter += 1

        r = tf.compat.v1.layers.Dropout(rate=self._rate)(
                tf.compat.v1.layers.Dense(
                    units=vshape[-3],
                    activation=None,
                    _reuse=tf.compat.v1.AUTO_REUSE,
                    name="l_final")(batched_features),
                training=self._train
            )

        ## output -> [h, r]

        r = tf.reshape(r, shape= vshape[:-2] + [1,1])

        # print(" r :" + str(r.shape))

        return r, None

    def _activation(self, s, c, votes, poses):
        ## poses :: { batch, output_atoms, new_w, new_h, 1 } + repdim
        ## votes :: { batch, output_atoms, new_w, new_h, depth * np.prod(ksizes) } + repdim
        ## c :: { batch, output_atoms, new_w , new_h, depth * np.prod(ksizes) }

        poses_tiled = tf.tile(poses, [1, 1, 1, 1, self.atoms, 1, 1])

        self._agreement = GaussianKernel().take(poses_tiled, votes)

        raw = tf.reduce_sum(tf.multiply(c, self._agreement), axis=-3, keepdims=True)

        if self._verbose:
            tf.compat.v1.summary.histogram(self.name + "dist_" + str(self._it), self._agreement)

        ## raw :: { batch, output_atoms, new_w, new_h, 1 }

        theta1 = variables.weight_variable([1], name="theta1", verbose=self._verbose)
        theta2 = variables.bias_variable([1], name="theta2", verbose=self._verbose)

        if self._activate:
            activation = tf.sigmoid(theta1 * raw + theta2)
        else:
            activation = theta1 * raw + theta2
        ## activation :: { batch, output_atoms, new_w, new_h, 1 }

        return activation
