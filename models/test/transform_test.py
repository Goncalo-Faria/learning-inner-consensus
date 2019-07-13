import tensorflow as tf

from models.coreimp.equiTransform import EquiTransform
from models.coreimp.commonMetrics import SquaredFrobenius

tf.compat.v1.enable_eager_execution()

batch = 32
w = 6
h = 3
depth = 3
representation_dim = [2, 2]
atoms = 4

t = EquiTransform(
    atoms,
    SquaredFrobenius()
)


poses = tf.ones([batch, w, h, depth] + representation_dim, dtype=tf.float32)
activations = tf.ones([batch, w, h, depth] , dtype=tf.float32)

high_votes, high_activations = t.translate(poses, activations)

print("got " + str(high_votes.shape))
print("should have been " + str([batch,atoms,w,h,depth] + representation_dim))

print("got " + str(high_activations.shape))
print("should have been " + str([batch,atoms,w,h,depth]))

