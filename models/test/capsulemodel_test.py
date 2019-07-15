import tensorflow as tf

from models.coreimp.kernelRouting import KernelRouting
from models.coreimp.commonKernels import DotProd
from models.coreimp.commonMetrics import Frobenius
from models.coreimp.equiTransform import EquiTransform
from models.layers.capsule import CapsuleLayer
from models.capsulemodel import CapsuleModel
from models.block.capsuleblock import CapsuleReductionBlock, CapsuleIdentityBlock


def setup_parameters_intricate(num_classes=10,max_steps = 1000,learning_rate = 0.001):

    hparams = {
        "learning_rate": learning_rate,
        "max_steps": max_steps,
        "loss_type": 'margin',
        "remake": False,
        "verbose": True,
        "regulizer_constant": 0.0,
        "derender_layers": [
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=[5, 5],
                activation='relu',
                use_bias=True,
                padding="VALID"
            )
        ],
        "primary_parameters": {
            "pose_dim": [4, 4],
            "ksize": 1,
            "groups": 32
        },
        "last_layer": {
            "transform": EquiTransform(
                output_atoms=num_classes,
                metric=Frobenius()
            ),
            "routing": KernelRouting(
                kernel=DotProd(),
                metric=Frobenius(),
                iterations=3
            )
        },
        "reconstruction_layer_sizes": [512, 1024],
        "layers": [
            CapsuleReductionBlock(
                routing = KernelRouting(
                    kernel=DotProd(),
                    metric=Frobenius(),
                    iterations=3
                ),
                metric=Frobenius(),
                stride=2,
                layer_sizes=[16,32,32],
                name = "initial_reduction"
            ),
            CapsuleIdentityBlock(
                routing=KernelRouting(
                    kernel=DotProd(),
                    metric=Frobenius(),
                    iterations=3
                ),
                metric=Frobenius(),
                layer_sizes=[16, 32, 32],
                name="keep"
            ),
            CapsuleReductionBlock(
                routing=KernelRouting(
                    kernel=DotProd(),
                    metric=Frobenius(),
                    iterations=3
                ),
                metric=Frobenius(),
                stride=4,
                layer_sizes=[32, 64, 64],
                name="final_reduction"
            ),
        ]
    }
    return hparams


def setup_parameters_trivial(num_classes=10,max_steps = 1000,learning_rate = 0.001):
    router = KernelRouting(
        DotProd(),
        Frobenius(),
        iterations=3
    )
    hparams = {
        "learning_rate": learning_rate,
        "max_steps": max_steps,
        "loss_type": 'margin',
        "remake": False,
        "verbose": True,
        "regulizer_constant": 0.0,
        "derender_layers": [
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=[5, 5],
                activation='relu',
                use_bias=True,
                padding="VALID"
            )
        ],
        "primary_parameters": {
            "pose_dim": [4, 4],
            "ksize": 1,
            "groups": 32
        },
        "last_layer": {
            "transform": EquiTransform(
                output_atoms=num_classes,
                metric=Frobenius()
            ),
            "routing": KernelRouting(
                kernel=DotProd(),
                metric=Frobenius(),
                iterations=3
    )
        },
        "reconstruction_layer_sizes": [512, 1024],
        "layers": [
            CapsuleLayer(
                routing=router,
                transform=EquiTransform(
                    output_atoms=32,
                    metric=Frobenius()
                ),
                ksizes=[1, 3, 3, 1]
            ),
            CapsuleLayer(
                routing=router,
                transform=EquiTransform(
                    output_atoms=32,
                    metric=Frobenius()
                ),
                ksizes=[1, 3, 3, 1]
            )
        ]
    }
    return hparams


features = {
    "images":  tf.ones(shape = [4,1,24,24], dtype=tf.float32),
    "depth":  1,
    "height": 24,
    "width":  24
}

result = CapsuleModel(setup_parameters_intricate(10)).inference(features)

print(result)