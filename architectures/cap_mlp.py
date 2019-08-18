import tensorflow as tf

from models.coreimp.commonMetrics import Frobenius
from models.coreimp.equiTransform import EquiTransform
from models.coreimp.rnnRouting import RNNRouting
from models.layers.capsule import CapsuleLayer


## smaller version.

def setup(
        hparams):

    router1 = RNNRouting(
        metric=Frobenius(),
        iterations=3,
        cell=tf.compat.v1.nn.rnn_cell.LSTMCell(
            num_units=16,
            name="attentionLayer1",
            reuse=tf.compat.v1.AUTO_REUSE),
        verbose=hparams.verbose,
        name="router1",
        bias=False
    )

    router2 = RNNRouting(
        metric=Frobenius(),
        iterations=3,
        cell=tf.compat.v1.nn.rnn_cell.LSTMCell(
            num_units=16,
            name="attentionLayer2",
            reuse=tf.compat.v1.AUTO_REUSE),
        verbose=hparams.verbose,
        name="router2",
        bias=False
    )

    router3 = RNNRouting(
        metric=Frobenius(),
        iterations=3,
        cell=tf.compat.v1.nn.rnn_cell.LSTMCell(
            num_units=16,
            name="attentionLayer3",
            reuse=tf.compat.v1.AUTO_REUSE),
        verbose=hparams.verbose,
        name="router3",
        bias=False
    )

    hparams.derender_layers= [
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=[5, 5],
                activation='relu',
                use_bias=True,
                padding="VALID",
                strides=[2,2],
                kernel_regularizer=tf.compat.v1.keras.regularizers.l2(0.0000002),
                bias_regularizer=tf.compat.v1.initializers.truncated_normal(mean=0.0, stddev=0.01)
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
            "routing" : router3
        }
    hparams.reconstruction_layer_sizes= [512, 1024]
    hparams.layers= [
            CapsuleLayer(
                routing= router2,
                transform=EquiTransform(
                    output_atoms=16,
                    metric=Frobenius(),
                name="ATransf"
                ),
                ksizes=[1, 3, 3, 1],
                strides=[1,2,2,1],
                name = "A"
            ),
            CapsuleLayer(
                routing= router1,
                transform=EquiTransform(
                    output_atoms=16,
                    metric=Frobenius(),
                name="BTransf"
                ),
                ksizes=[1, 3, 3, 1],
                name= "B"
            )
        ]

    return hparams