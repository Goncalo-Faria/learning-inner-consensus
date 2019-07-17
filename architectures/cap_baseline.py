import tensorflow as tf

from models.coreimp.kernelRouting import KernelRoutingWithPrior
from models.coreimp.commonKernels import DotProd
from models.coreimp.commonMetrics import Frobenius, SquaredFrobenius
from models.coreimp.equiTransform import EquiTransform
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
                metric=SquaredFrobenius()
            ),
            "routing" : KernelRoutingWithPrior(
                kernel=DotProd(),
                metric=SquaredFrobenius(),
                iterations=3,
                verbose=True,
                name="LastR"
            )
        }
    hparams.reconstruction_layer_sizes= [512, 1024]
    hparams.layers= [
            CapsuleLayer(
                routing= KernelRoutingWithPrior(
                    DotProd(),
                    SquaredFrobenius(),
                    iterations=3,
                    verbose=True,
                    name="A"
                ),
                transform=EquiTransform(
                    output_atoms=16,
                    metric=SquaredFrobenius()
                ),
                ksizes=[1, 3, 3, 1],
                strides=[1,2,2,1],
                name = "A"
            ),
            CapsuleLayer(
                routing=KernelRoutingWithPrior(
                    DotProd(),
                    SquaredFrobenius(),
                    iterations=3,
                    verbose=True,
                    name="B"
                ),
                transform=EquiTransform(
                    output_atoms=16,
                    metric=SquaredFrobenius()
                ),
                ksizes=[1, 3, 3, 1],
                name= "B"
            )
        ]

    return hparams
