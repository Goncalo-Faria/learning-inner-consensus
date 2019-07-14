import tensorflow as tf

from coreimp.kernelRouting import KernelRouting
from coreimp.commonKernels import DotProd
from coreimp.commonMetrics import Frobenius
from coreimp.equiTransform import EquiTransform
from layers.capsule import CapsuleLayer
from capsulemodel import CapsuleModel


def setup_parameters(num_classes=10):
    router = KernelRouting(
        DotProd(),
        Frobenius(),
        iterations=3
    )
    hparams = {
        "learning_rate": 0.001,
        "decay_rate": 0.96,
        "decay_steps": 2000,
        "loss_type": 'margin',
        "remake": False,
        "verbose": True,
        "regulizer_constant": 0.0,
        "derender_layers": [
            tf.keras.layers.Conv2D(
                filters=128,
                kernel_size=[9, 9],
                activation='relu',
                use_bias=True,
                padding="VALID"
            )
        ],
        "primary_parameters": {
            "pose_dim": [4, 4],
            "ksize": 5,
            "groups": 32
        },
        "last_layer": {
            "transform": EquiTransform(
                output_atoms=num_classes,
                metric=Frobenius()
            ),
            "routing": router
        },
        "reconstruction_layer_sizes": [512, 1024],
        "layers": [
            CapsuleLayer(
                routing=router,
                transform=EquiTransform(
                    output_atoms=32,
                    metric=Frobenius()
                ),
                ksizes=[1, 5, 5, 1]
            ),
            CapsuleLayer(
                routing=router,
                transform=EquiTransform(
                    output_atoms=32,
                    metric=Frobenius()
                ),
                ksizes=[1, 5, 5, 1]
            )
        ]
    }
    return hparams



print("before")

result = CapsuleModel(setup_parameters(10)).inference( tf.ones(shape = [4,24,24,1], dtype=tf.float32))

print(result)