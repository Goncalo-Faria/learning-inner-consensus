import tensorflow as tf

from ..coreimp.equiTransform import EquiTransform
from ..layers.capsule import CapsuleLayer


class CapsuleIdentityBlock(object):
    def __init__(self,
                 routing,
                 metric,
                 layer_sizes=[32,64,64],
                 name = ""
                 ):

        assert( len(layer_sizes) == 3 ) , " layer sizes must have len 3. "

        self.name = "CapsuleIdentityBlock/" + name

        self._A = CapsuleLayer(
            routing=routing,
            iterations = 1,
            transform = EquiTransform(
                output_atoms=layer_sizes[0],
                metric=metric
            ),
            ksizes=[1, 1, 1, 1],
            name="stage1"
        )

        self._B = CapsuleLayer(
            routing=routing,
            iterations = 3,
            transform = EquiTransform(
                output_atoms=layer_sizes[1],
                metric=metric
            ),
            padding="SAME",
            ksizes=[1, 3, 3, 1],
            name="stage2"
        )

        self._C = CapsuleLayer(
            routing=routing,
            iterations=1,
            transform=EquiTransform(
                output_atoms=layer_sizes[2],
                metric=metric
            ),
            ksizes=[1, 1, 1, 1],
            name="stage3"
        )

    def inference(self, input_tensor):

        with tf.compat.v1.variable_scope(self.name,reuse=tf.compat.v1.AUTO_REUSE):
            X = self._A.inference(input_tensor)
            X = self._B.inference(X)
            X = self._C.inference(X)

            x_poses, x_activations = X
            poses, activations = input_tensor

            activations = tf.concat(axis=3, values=[x_activations, activations])
            poses = tf.concat(axis=3, values=[x_poses, poses])

            return poses, activations
