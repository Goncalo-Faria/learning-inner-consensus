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
                metric=Frobenius()
            ),
            "routing" : KernelRouting(
                kernel= MonoKernelMix(
                    kernel=SpectralMixture(hparams.verbose),
                    degree=10),
                metric=Frobenius(),
                iterations=3,
                verbose=hparams.verbose,
                name="LastR",
                activate = False,
            )
        }
    hparams.reconstruction_layer_sizes= [512, 1024]
    hparams.layers= [
            CapsuleLayer(
                routing= KernelRouting(
                    kernel=MonoKernelMix(
                        kernel=SpectralMixture(hparams.verbose),
                        degree=10),
                    metric=Frobenius(),
                    iterations=3,
                    verbose=hparams.verbose,
                    name="A"
                ),
                transform=EquiTransform(
                    output_atoms=16,
                    metric=Frobenius()
                ),
                ksizes=[1, 3, 3, 1],
                strides=[1,2,2,1],
                name = "A"
            ),
            CapsuleLayer(
                routing=KernelRouting(
                    kernel=MonoKernelMix(
                        kernel=SpectralMixture(hparams.verbose),
                        degree=10),
                    metric=Frobenius(),
                    iterations=3,
                    verbose=hparams.verbose,
                    name="B"
                ),
                transform=EquiTransform(
                    output_atoms=16,
                    metric=Frobenius()
                ),
                ksizes=[1, 3, 3, 1],
                name= "B"
            )
        ]

    return hparams