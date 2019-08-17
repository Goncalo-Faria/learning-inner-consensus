from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from models.core.routing import HyperSimplifiedRoutingProcedure


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

        print("vote : " + str(votes.shape) )
        print("activations : " + str(activations.shape) )

        votes_flatten = tf.reshape(votes, shape=vshape[:-2] + [-1])

        activations_flatten = tf.reshape(activations, shape=vshape[:-2] + [-1])

        capsule_flatten = tf.concat([votes_flatten, activations_flatten], axis=-1)

        local_capsules_flatten = tf.reshape(capsule_flatten, shape=vshape[:-3] + [-1])

        batched_features = tf.reshape(local_capsules_flatten, [-1, local_capsules_flatten.shape[-1]])

        counter = 0

        for layer_num in self._compatibility_layers:
            print("batched_features: " + str(batched_features.shape))
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

        s = batched_features

        print("batched_features: " + str(batched_features.shape))

        r = tf.compat.v1.layers.Dropout(rate=self._rate)(
                tf.compat.v1.layers.Dense(
                    units=vshape[-3],
                    activation=None,
                    _reuse=tf.compat.v1.AUTO_REUSE,
                    name="l_final")(batched_features),
                training=self._train
            )

        print("batched_features: " + str(r.shape))
        ## output -> [h, r]

        r = tf.reshape(r, shape= vshape[:-2] + [1,1])

        # print(" r :" + str(r.shape))

        return r, s

    def _activation(self, s, c, votes, poses):
        ## s :: batch, output_atoms, new_w, new_h, 1] + [degree]

        counter = 0

        batched_features = s

        for layer_num in self._activation_layers :
            batched_features = tf.compat.v1.layers.Dropout(rate=self._rate)(
                    tf.compat.v1.layers.Dense(
                        units=layer_num,
                        activation=tf.nn.relu,
                        _reuse=tf.compat.v1.AUTO_REUSE,
                        name="l_"+str(counter))(batched_features),
                    training=self._train
                )

            counter += 1
        ## apply nn

        if self._activate :
            activation = tf.compat.v1.layers.Dropout(rate=self._rate)(
                    tf.compat.v1.layers.Dense(
                        units=1,
                        name="l_final",
                        _reuse=tf.compat.v1.AUTO_REUSE,
                        activation=tf.nn.sigmoid
                    )(batched_features),
                    training=self._train
            )
        else :
            activation = tf.compat.v1.layers.Dropout(rate=self._rate)(
                    tf.compat.v1.layers.Dense(
                        units=1,
                        name="l_final_logits",
                        _reuse=tf.compat.v1.AUTO_REUSE
                    )(batched_features),
                    training=self._train
            )

        ## activation :: { batch, output_atoms, new_w, new_h, 1 }
        activation = tf.reshape(activation,votes.shape.as_list()[:-3]+[1,1,1])

        # print( "activation" + str(activation.shape))

        return activation ## batch , out , w, h, 1, 1
