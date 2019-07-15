from ..layers.capsule import CapsuleLayer
from ..coreimp.equiTransform import EquiTransform
import tensorflow as tf


class CapsuleIdentityBlock(object):
    def __init__(self,
                 routing,
                 metric,
                 layer_sizes=[32,64,64],
                 name = ""
                 ):

        assert( len(layer_sizes) == 3 ) , " layer sizes must have len 3. "

        self.name = name

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

        with tf.name_scope(self.name):
            X = self._A.inference(input_tensor)
            X = self._B.inference(X)
            X = self._C.inference(X)

            x_poses, x_activations = X
            poses, activations = input_tensor

            activations = tf.concat(axis=3, values=[x_activations, activations])
            poses = tf.concat(axis=3, values=[x_poses, poses])

            return poses, activations


class CapsuleReductionBlock(object):
    def __init__(self,
                 routing,
                 metric,
                 stride,
                 name="",
                 layer_sizes=[32,64,64]
                 ):

        assert( len(layer_sizes) == 3 ) , " sizes must have len 3. "
        self.name = name

        self._A = CapsuleLayer(
            routing=routing,
            iterations = 1,
            transform = EquiTransform(
                output_atoms=layer_sizes[0],
                metric=metric
            ),
            ksizes=[1, 1, 1, 1],
            strides = [1,stride,stride,1],
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

        self._sharedtransform = EquiTransform(
                output_atoms=layer_sizes[2],
                metric=metric)

        self._C = CapsuleLayer(
            routing=routing,
            iterations=1,
            transform = self._sharedtransform,
            ksizes=[1, 1, 1, 1],
            name="stage3A"
        )

        self._C_side = CapsuleLayer(
            routing=routing,
            iterations=1,
            transform=self._sharedtransform,
            ksizes=[1, 1, 1, 1],
            strides=[1, stride, stride, 1],
            name="stage3B"
        )

    def inference(self, input_tensor):
        with tf.name_scope(self.name):
            X = self._A.inference(input_tensor)
            X = self._B.inference(X)
            X = self._C.inference(X)

            x_poses, x_activations = X
            poses, activations = self._C_side.inference(input_tensor)

            activations = tf.concat(axis=3, values=[x_activations, activations])
            poses  = tf.concat(axis=3, values=[x_poses, poses])

            return poses, activations
