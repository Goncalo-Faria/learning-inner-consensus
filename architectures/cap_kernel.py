import tensorflow as tf

from models.coreimp.commonKernels import SpectralMixture, GaussianKernel
from models.coreimp.commonMetrics import Frobenius
from models.coreimp.equiTransform import EquiTransform
from models.coreimp.kernelRouting import KernelRouting
from models.coreimp.kernelmix import MonoKernelMix
from models.layers.capsule import CapsuleLayer


## smaller version.

def setup(
        hparams, k):
    router1 = KernelRouting(
                kernel = MonoKernelMix(
                    kernel=k,
                    degree=4),
                metric=Frobenius(),
                iterations=3,
                verbose=hparams.verbose,
                name="router1"
    )

    router2 = KernelRouting(
        kernel=MonoKernelMix(
            kernel=k,
            degree=4),
        metric=Frobenius(),
        iterations=3,
        verbose=hparams.verbose,
        name="router2"
    )

    router3 = KernelRouting(
        kernel=MonoKernelMix(
            kernel=k,
            degree=3),
        metric=Frobenius(),
        iterations=12,
        verbose=hparams.verbose,
        name="router3"
    )

    hparams.derender_layers= [
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=[5, 5],
                activation='relu',
                use_bias=True,
                padding="VALID",
                strides=[2,2],
                bias_initializer=tf.compat.v1.initializers.truncated_normal(mean=0, stddev=0.01)
            )
        ]
    hparams.primary_parameters= {
            "pose_dim": [4, 4],
            "ksize": 1,
            "groups" : 8
        }
    hparams.last_layer= {
            "transform": EquiTransform(
                output_atoms=hparams.num_classes,
                metric=Frobenius(),
                name="FTransf"
            ),
            "routing": router3
        }
    hparams.reconstruction_layer_sizes=[512, 1024]
    hparams.layers= [
            CapsuleLayer(
                routing=router1,
                transform=EquiTransform(
                    output_atoms=16,
                    metric=Frobenius(),
                    name="ATransf"
                ),
                ksizes=[1, 3, 3, 1],
                strides=[1, 2, 2, 1],
                name="A",
                coordinate_addition=True
            ),
            CapsuleLayer(
                routing=router2,
                transform=EquiTransform(
                    output_atoms=16,
                    metric=Frobenius(),
                    name="BTransf"
                ),
                ksizes=[1, 3, 3, 1],
                name="B",
                coordinate_addition=True
            )
        ]

    return hparams
