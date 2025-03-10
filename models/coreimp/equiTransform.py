from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ..core import variables
from ..core.metric import Metric
from ..core.transform import Transform
from ..util.initializer import IdentityRandomUniform

from opt_einsum import contract

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
            initializer = IdentityRandomUniform(
                    maxval=0.1, minval=-0.1, dtype=tf.float32)
        )

        #print("W_raw mat: " + str(W.shape) )
        ## W :: {1, outputatoms, 1 , 1, depth * np.prod(ksizes) } + repdim

        W_norm = self.metric.take(W)
        ## W_norm :: { 1, outputatoms, 1, 1, depth * np.prod(ksizes) , 1 ,1 }

        votes_raw = contract("aocduik,bwhukj->bowhuij", W, poses)

        activations_tiled = tf.tile(
            tf.expand_dims(activations, 1),
            [1, self._output_atoms, 1, 1, 1, 1, 1]
        )
        ## activations_tiled :: { batch, outputatoms, new_w, new_h, depth * np.prod(ksizes) }

        votes = votes_raw / (W_norm + self._epsilon)

        """
            transforms the lower level poses to the higher level capsule space.
        """
        ## votes :: { batch, outputatoms, new_w, new_h, depth * np.prod(ksizes) } + repdim
        #print("votes: " + str(votes.shape))
        return votes, activations_tiled
