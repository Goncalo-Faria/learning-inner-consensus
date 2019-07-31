from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ..core import variables
from ..core.transform import Transform
from ..core.metric import Metric
from ..util.initializer import IdentityRandomUniform


class EquiTransform(Transform):

    def __init__(
            self,
            output_atoms,
            metric,
            verbose=False,
            epsilon=1e-6,
            name=""):
        self.metric = metric
        self._epsilon = epsilon

        assert isinstance(metric, Metric), \
            " metric must be instance of Metric metaclass. "

        super(EquiTransform, self).__init__(
            "EquiTransform" + name,
            output_atoms,
            verbose)

    def apply(self, poses, activations):
        ## poses { batch, new_w , new_h, depth * np.prod(ksizes) } + repdim
        ## activations { batch, new_w , new_h, depth * np.prod(ksizes) }

        poses_shape = poses.shape.as_list()

        W = variables.weight_variable(
            [1, self._output_atoms, 1, 1, poses_shape[-3], poses_shape[-2], poses_shape[-1]],
            name="scale",
            initializer = IdentityRandomUniform(
                    maxval=0.1, minval=-0.1, dtype=tf.float32)
        )

        ## W :: {1, outputatoms, 1 , 1, depth * np.prod(ksizes) } + repdim

        W_norm = self.metric.take(W)
        ## W_norm :: { 1, outputatoms, 1, 1, depth * np.prod(ksizes) , 1 ,1 }

        W_scaled = tf.tile(
            W,
            [poses_shape[0], 1, poses_shape[1], poses_shape[2], 1, 1, 1]
        )
        ## W_scaled :: {batch, outputatoms, new_w , new_h, depth * np.prod(ksizes) } + repdim

        W_norm_scaled = tf.tile(
            W_norm,
            [poses_shape[0], 1, poses_shape[1], poses_shape[2], 1, poses_shape[4], poses_shape[5]]
        )
        ## W_norm_scaled :: { batch, outputatoms, new_w, new_h, depth * np.prod(ksizes) } + repdim

        poses_tiled = tf.tile(
            tf.expand_dims(poses, 1),
            [1, self._output_atoms, 1, 1, 1, 1, 1]
        )
        ## poses_tiled :: { batch, outputatoms, new_w, new_h, depth * np.prod(ksizes) } + repdim

        activations_tiled = tf.tile(
            tf.expand_dims(activations, 1),
            [1, self._output_atoms, 1, 1, 1]
        )
        ## activations_tiled :: { batch, outputatoms, new_w, new_h, depth * np.prod(ksizes) }

        votes = tf.matmul(W_scaled, poses_tiled) / (W_norm_scaled + self._epsilon)

        """
            transforms the lower level poses to the higher level capsule space.
        """
        ## votes :: { batch, outputatoms, new_w, new_h, depth * np.prod(ksizes) } + repdim

        return votes, activations_tiled
