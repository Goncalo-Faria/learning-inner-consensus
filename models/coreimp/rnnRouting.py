from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from models.core.routing import SimplifiedRoutingProcedure


class RNNRouting(SimplifiedRoutingProcedure):

    def __init__(
            self,
            metric,
            iterations,
            cell,
            bias=False,
            name="",
            epsilon=1e-6,
            compatibility_layers=[],
            activation_layers=[],
            normalization=tf.nn.softmax,
            train = False,
            verbose=False):
        self._compatibility_layers = compatibility_layers
        self._activation_layers = activation_layers
        self._cell = cell
        self.train = train

        super(RNNRouting, self).__init__(
            name="RNNRouting_" + name,
            metric=metric,
            design_iterations=iterations,
            initial_state=None,
            verbose=verbose,
            bias=bias,
            epsilon=epsilon,
            normalization=normalization)


    def _compatibility(self, s, r, votes, poses, probabilities, activations, it):
        ## poses :: { batch, output_atoms, new_w, new_h, 1 } + repdim
        ## votes :: { batch, output_atoms, new_w, new_h, depth * np.prod(ksizes) } + repdim
        ## r :: { batch, output_atoms, new_w , new_h, depth * np.prod(ksizes) }

        vshape = votes.shape.as_list()
        # s :: {degree}

        poses_tiled = tf.tile(poses, [1, 1, 1, 1, vshape[4], 1, 1])
        poses_concat = tf.reshape(poses_tiled, poses_tiled.shape.as_list()[:-2]+[1]+[-1])
        votes_concat = tf.reshape(votes, votes.shape.as_list()[:-2]+[1]+[-1])

        #s_tile = tf.tile(s, [1, 1, 1, 1, vshape[4], 1])

        #s_concat = tf.expand_dims(s_tile, axis=-2)

        #  concat( h : mu: a: v : r)(degree + 16 + 1 + 16 + 1 )

        stacked_values = tf.concat([poses_concat, activations, votes_concat, r], axis=-1)

        flatten_stacked_values = tf.reshape(stacked_values, [-1, stacked_values.shape.as_list()[-1]] )

        inl = flatten_stacked_values
        #
        if s is None:
            s = self._cell.get_initial_state(
                inputs=inl,
                batch_size=inl.shape.as_list()[0],
                dtype=tf.float32)

        out, s = self._cell(
            inputs=inl,
            state=s)

        feature_map = s[0]

        counter = 0

        for layer_num in self._compatibility_layers:
            feature_map = tf.compat.v1.layers.Dense(
                    units=layer_num,
                    activation=tf.nn.relu,
                    _reuse=tf.compat.v1.AUTO_REUSE,
                    name="l_" + str(counter)
                )(feature_map)
            ## apply nn

            counter += 1

        outl = tf.compat.v1.layers.Dense(
                units=1,
                activation=None,
                _reuse=tf.compat.v1.AUTO_REUSE,
                name="l_final"
            )(feature_map)

        r = tf.reshape(outl, vshape[0:5]+[1,1])

        # final

        ## s :: { batch, output_atoms, new_w , new_h, depth * np.prod(ksizes), degree }
        ## r :: { batch, output_atoms, new_w , new_h, depth * np.prod(ksizes), 1 }
        c = self._normalization(r, axis=4)

        return c, s


    def _activation(self, s, c, votes, poses):
     ## votes :: { batch, output_atoms, new_w, new_h, depth * np.prod(ksizes) } + repdim
        vshape = votes.shape.as_list()

        degree = s[0].shape[-1]

        new_c = tf.reshape(s[1], vshape[0:-2]+[1,degree])

        combined_c = tf.reduce_sum(new_c * c, axis=-3, keepdims=True)

        inl = tf.reshape(combined_c, [-1, degree])

        counter = 0

        for layer_num in self._activation_layers:
            inl = tf.compat.v1.layers.Dense(
                    units=layer_num,
                    activation=tf.nn.relu,
                    _reuse=tf.compat.v1.AUTO_REUSE,
                    name="l_" + str(counter)
                )(inl)
            ## apply nn

            counter += 1

        if self._activate :
            outl = tf.compat.v1.layers.Dense(
                    units=1,
                    name="l_final",
                    _reuse=tf.compat.v1.AUTO_REUSE,
                    activation=tf.nn.sigmoid
                )(inl)
        else :
            outl = tf.compat.v1.layers.Dense(
                    units=1,
                    name="l_final_logits",
                    _reuse=tf.compat.v1.AUTO_REUSE
                )(inl)


        ## activation :: { batch, output_atoms, new_w, new_h, 1 }
        activation = tf.reshape(outl,vshape[:-3]+[1,1,1])

        return activation ## batch , out , w, h, 1, 1


