from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from models.core.routing import RoutingProcedure

class NiNRouting(RoutingProcedure):

    def __init__(
            self,
            metric,
            iterations,
            activation_layers=[],
            compatibility_layers=[],
            degree=16,
            activate = True,
            name="",
            verbose=False):
        self._degree = degree
        self._activation_layers = activation_layers
        self._compatibility_layers = compatibility_layers

        super(NiNRouting, self).__init__(
            name="NiNRouting" + name,
            metric=metric,
            design_iterations=iterations,
            initial_state=None,
            activate=activate,
            verbose=verbose)

    def _compatibility(self, s, r, votes, poses, probabilities, activations, it):
        ## poses :: { batch, output_atoms, new_w, new_h, 1 } + repdim
        ## votes :: { batch, output_atoms, new_w, new_h, depth * np.prod(ksizes) } + repdim
        ## r :: { batch, output_atoms, new_w , new_h, depth * np.prod(ksizes) }

        vshape = votes.shape.as_list()
        # s :: {degree}

        if s is None:
            s = self._cached_h

        poses_tiled = tf.tile(poses, [1, 1, 1, 1, vshape[4], 1, 1])
        poses_concat = tf.reshape(poses_tiled, poses_tiled.shape.as_list()[:-2]+[1]+[-1])
        votes_concat = tf.reshape(votes, votes.shape.as_list()[:-2]+[1]+[-1])
        s_concat = tf.expand_dims( s, axis=-2)

        ##  concat( h : mu: a: v : r)( degree + 16 + 1 + 16 + 1 )

        stacked_values = tf.concat([s_concat,poses_concat, activations, votes_concat, r], axis=-1)

        flatten_stacked_values = tf.reshape(stacked_values, [-1, stacked_values.shape.as_list()[-1]] )

        inl = flatten_stacked_values

        for layer_num in self._compatibility_layers:

            inl = tf.compat.v1.layers.Dense(
                layer_num,
                activation=tf.nn.relu)(inl)
        ## apply nn

        outl = tf.compat.v1.layers.Dense(
                self._degree + 1,
                activation=None)(inl)

        ## output -> [h, r]
        s = outl[:,:-1]
        r = outl[:,-1]

        s = tf.reshape(s, vshape[0:5]+[self._degree])
        r = tf.reshape(r, vshape[0:5]+[1,1])

        # final

        ## s :: { batch, output_atoms, new_w , new_h, depth * np.prod(ksizes), degree }
        ## r :: { batch, output_atoms, new_w , new_h, depth * np.prod(ksizes), 1 }

        return r, s


    def _activation(self, s, c, votes, poses):
        ## s :: batch, output_atoms, new_w, new_h, depth * np.prod(ksizes)]

        vshape = votes.shape.as_list()

        if s is None :
            s = tf.zeros([1, 1, 1, 1, 1, self._degree], dtype=tf.float32)
            s = tf.tile(s, [vshape[0], vshape[1], vshape[2], vshape[3], vshape[4], 1])
            self._cached_h = s

        sshape = s.shape.as_list()

        inl = tf.reshape( s,[sshape[0]*sshape[1]*sshape[2]*sshape[3], -1])

        for layer_num in self._activation_layers :

            inl = tf.compat.v1.layers.Dense(
                layer_num,
                activation=tf.nn.relu)(inl)
        ## apply nn

        outl = tf.compat.v1.layers.Dense(
                1,
                activation=tf.nn.sigmoid)(inl)

        ## activation :: { batch, output_atoms, new_w, new_h, 1 }
        activation = tf.reshape(outl,sshape[:-2]+[1,1,1])

        return activation ## batch , out , w, h, 1, 1

    def _initial_coefficients(self, r, activations):
        return tf.ones(activations.shape.as_list(), dtype=tf.float32) / activations.shape.as_list()[-1]

