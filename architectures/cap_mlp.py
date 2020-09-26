import tensorflow as tf

from models.coreimp.commonMetrics import Frobenius
from models.coreimp.equiTransform import EquiTransform
from models.coreimp.rnnRouting import RNNRouting
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
                strides=[2,2],
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
                output_atoms = hparams.num_classes,
                metric=Frobenius(),
                name="FTransf"
            ),
            "routing" : RNNRouting(
                metric=Frobenius(),
                iterations=3,
                cell=tf.compat.v1.nn.rnn_cell.LSTMCell(
                    num_units=hparams.degree,
                    name="attentionLayer3"),
                verbose=hparams.verbose,
                name="router3",
                bias=False,
                compatibility_layers=[64,64],
                activation_layers=[124,124],
                train=hparams.train
            )
        }
    hparams.reconstruction_layer_sizes= [512, 1024]
    hparams.layers= [
            CapsuleLayer(
                routing= RNNRouting(
                    metric=Frobenius(),
                    iterations=3,
                    cell=tf.compat.v1.nn.rnn_cell.LSTMCell(
                        num_units=hparams.degree,
                        name="attentionLayer1"),
                    verbose = hparams.verbose,
                    name="router1",
                bias=False,
                compatibility_layers=[],
                activation_layers=[],
                train=hparams.train
                ),
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
                routing= RNNRouting(
                    metric=Frobenius(),
                    iterations=3,
                    cell=tf.compat.v1.nn.rnn_cell.LSTMCell(
                        num_units=hparams.degree,
                        name="attentionLayer2"),
                    verbose=hparams.verbose,
                    name="router2",
                    bias=False,
                    compatibility_layers=[32,32],
                    activation_layers=[64,64],
                    train=hparams.train
            ),
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
