import tensorflow as tf

from models.coreimp.kernelRouting import KernelRouting
from models.coreimp.commonKernels import DotProd
from models.coreimp.commonMetrics import Frobenius
from models.coreimp.equiTransform import EquiTransform
from models.block.capsuleblock import CapsuleIdentityBlock
from models.layers.capsule import CapsuleLayer


def setup(
        hparams):
    hparams.derender_layers=[
        tf.keras.layers.Conv2D(
            filters=32,
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
        "groups": 32
    }
    hparams.last_layer= {
        "transform": EquiTransform(
            output_atoms=hparams.num_classes,
            metric=Frobenius()
        ),
        "routing": KernelRouting(
            kernel=DotProd(),
            metric=Frobenius(),
            iterations=3,
            verbose=hparams.verbose,
            name="D"
        )
    }
    hparams.reconstruction_layer_sizes= [512, 1024]
    hparams.layers=[
        CapsuleLayer(
            routing = KernelRouting(
                kernel = DotProd(),
                metric = Frobenius(),
                iterations = 1,
                verbose=hparams.verbose,
                name="A"
            ),
            transform=EquiTransform(
                output_atoms=8,
                metric=Frobenius()
            ),
            ksizes=[1, 3, 3, 1],
            strides=[1, 2, 2, 1],
            name="initial_reduction"
        ),
        CapsuleIdentityBlock(
            routing=KernelRouting(
                kernel=DotProd(),
                metric=Frobenius(),
                iterations=3,
                verbose=hparams.verbose,
                name="B"
            ),
            metric=Frobenius(),
            layer_sizes=[8, 16, 16],
            name="keep"
        ),
        CapsuleLayer(
            routing=KernelRouting(
                kernel=DotProd(),
                metric=Frobenius(),
                iterations=3,
                verbose=hparams.verbose,
                name="C"
            ),
            transform=EquiTransform(
                output_atoms=16,
                metric=Frobenius()
            ),
            ksizes=[1, 1, 1, 1],
            strides=[1, 1, 1, 1],
            name="final_reduction"
        ),
    ]
    return hparams