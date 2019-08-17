import tensorflow as tf


from models.coreimp.commonMetrics import Frobenius
from models.coreimp.equiTransform import EquiTransform
from models.coreimp.ninRouting import NiNRouting
from models.layers.capsule import CapsuleLayer

## smaller version.

def setup(
        hparams):
    router = NiNRouting(
        metric=Frobenius(),
        verbose = hparams.verbose,
        name="router",
        compatibility_layers=[512,512],
        activation_layers=[264],
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
            "routing" : router
        }
    hparams.reconstruction_layer_sizes= [512, 1024]
    hparams.layers= [
            CapsuleLayer(
                routing= router,
                transform=EquiTransform(
                    output_atoms=16,
                    metric=Frobenius(),
                name="ATransf"
                ),
                ksizes=[1, 3, 3, 1],
                strides=[1,2,2,1],
                name = "A",
                coordinate_addition=True
            ),
            CapsuleLayer(
                routing= router,
                transform=EquiTransform(
                    output_atoms=16,
                    metric=Frobenius(),
                name="BTransf"
                ),
                ksizes=[1, 3, 3, 1],
                name= "B",
                coordinate_addition=True
            )
        ]

    return hparams