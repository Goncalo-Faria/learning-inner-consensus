from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ..core.routing import SimplifiedRoutingProcedure
from ..coreimp.commonMetrics import SquaredFrobenius
from ..coreimp.commonKernels import DotProd

class DynamicRouting(SimplifiedRoutingProcedure):

    def __init__(
            self,
            iterations,
            name="",
            verbose=False):
        self._agreement = None

        super(DynamicRouting, self).__init__(
            name="DynamicRouting" + name,
            metric=SquaredFrobenius(),
            design_iterations=iterations,
            initial_state=None,
            verbose=verbose)

    def _compatibility(self, s, r, votes, poses, probabilities, activations, it):
        ## poses :: { batch, output_atoms, new_w, new_h, 1 } + repdim
        ## votes :: { batch, output_atoms, new_w, new_h, depth * np.prod(ksizes) } + repdim
        ## r :: { batch, output_atoms, new_w , new_h, depth * np.prod(ksizes) }
        ## r :: { batch, output_atoms, new_w , new_h, depth * np.prod(ksizes) }

        poses_norm = self.metric.take(poses)

        poses = poses / (tf.sqrt(poses_norm + self._epsilon)+self._epsilon)

        self._wj = poses_norm / (1 + poses_norm)

        poses_tiled = tf.tile(poses, [1, 1, 1, 1, self.atoms, 1, 1])

        r = self._r + self._wj * DotProd().take(poses_tiled, votes)

        self._r = r

        c = self._normalization(r, axis=4)

        return c, s

    def _initial_coefficients(self,activations):

        r = tf.zeros(shape= activations.shape,
                    dtype=tf.float32,
                    name="compatibility_value")

        self._r = r

        #c = self._normalization(r, axis=4)

        return activations

    def _activation(self, s, c, votes, poses):
        ## poses :: { batch, output_atoms, new_w, new_h, 1 } + repdim
        ## votes :: { batch, output_atoms, new_w, new_h, depth * np.prod(ksizes) } + repdim
        ## c :: { batch, output_atoms, new_w , new_h, depth * np.prod(ksizes) }

        return self._wj
