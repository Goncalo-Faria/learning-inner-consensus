import tensorflow as tf


from models.coreimp.commonMetrics import Frobenius
from models.coreimp.equiTransform import EquiTransform
from models.coreimp.rnnRouting import RNNRouting
from models.layers.capsule import CapsuleLayer

## smaller version.

def setup(
        hparams):

    midrouter = RNNRouting(
        metric=Frobenius(),
        iterations=3,
        degree=16,
        verbose=hparams.verbose,
        name="globalrouter",
        activate=True,
    )

    lastrouter = RNNRouting(
        metric=Frobenius(),
        iterations=3,
        degree=16,
        verbose=hparams.verbose,
        name="globalrouter",
        activate=False,
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
                metric=Frobenius()
            ),
            "routing" : lastrouter
        }
    hparams.reconstruction_layer_sizes= [512, 1024]
    hparams.layers= [
            CapsuleLayer(
                routing= midrouter,
                transform=EquiTransform(
                    output_atoms=16,
                    metric=Frobenius()
                ),
                ksizes=[1, 3, 3, 1],
                strides=[1,2,2,1],
                name = "A"
            ),
            CapsuleLayer(
                routing= midrouter,
                transform=EquiTransform(
                    output_atoms=16,
                    metric=Frobenius()
                ),
                ksizes=[1, 3, 3, 1],
                name= "B"
            )
        ]

    return hparams