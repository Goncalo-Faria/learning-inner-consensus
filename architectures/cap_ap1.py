import tensorflow as tf

from models.coreimp.kernelRouting import KernelRouting
from models.coreimp.commonKernels import SpectralMixture
from models.coreimp.commonMetrics import Frobenius, SquaredFrobenius
from models.coreimp.equiTransform import EquiTransform
from models.coreimp.kernelmix import MonoKernelMix
from models.layers.capsule import CapsuleLayer

## smaller version.

def setup(
        hparams):
    router = KernelRouting(
                kernel=MonoKernelMix(
                    kernel=SpectralMixture(hparams.verbose),
                    degree=4),
                metric=Frobenius(),
                iterations=3,
                verbose=hparams.verbose,
                name="globalRouter"
    )

    hparams.derender_layers= [
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=[5, 5],
                activation='relu',
                use_bias=True,
                padding="VALID",
                strides=[2,2]
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
            "routing": router
        }
    hparams.reconstruction_layer_sizes=[512, 1024]
    hparams.layers= [
            CapsuleLayer(
                routing=router,
                transform=EquiTransform(
                    output_atoms=16,
                    metric=Frobenius(),
                    name="ATransf"
                ),
                ksizes=[1, 3, 3, 1],
                strides=[1, 2, 2, 1],
                name="A"
            ),
            CapsuleLayer(
                routing=router,
                transform=EquiTransform(
                    output_atoms=16,
                    metric=Frobenius(),
                    name="BTransf"
                ),
                ksizes=[1, 3, 3, 1],
                name="B"
            )
        ]

    return hparams
