import tensorflow as tf


from models.capsulemodel import CapsuleModel
from architectures import cap_mlp as MLPCap
from argparse import Namespace

def setup(num_classes=10,max_steps = 1000,learning_rate = 0.001):

    hparams = Namespace()

    hparams.__dict__ = {
        "model" : "YourMUM",
        "learning_rate": learning_rate,
        "max_steps": max_steps,
        "num_classes":num_classes,
        "loss_type": 'margin',
        "remake": False,
        "verbose": True,
        "regulizer_constant": 0.0
    }
    return hparams


features = {
    "images":  tf.ones(shape = [4,1,24,24], dtype=tf.float32),
    "depth":  1,
    "height": 24,
    "width":  24
}

result = CapsuleModel(MLPCap.setup(setup(10))).inference(features)

print(result)