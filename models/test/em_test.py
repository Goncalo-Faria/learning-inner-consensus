import tensorflow as tf

from models.coreimp.emRouting import EMRouting
from models.coreimp.commonMetrics import Frobenius


tf.compat.v1.enable_eager_execution()

batch = 32
w = 5
h = 5
depth = 16
representation_dim = [2, 2]
atoms = 4

print("batch:" + str(batch))
print("w:" + str(w))
print("h:" + str(h))
print("depth:" + str(depth))
print("repdim:" + str(representation_dim))
print("out:" + str(atoms) )

r = EMRouting(
    metric=Frobenius(),
    iterations=3,
    name="test"
)


votes = tf.ones([batch, atoms, w, h, depth] + representation_dim, dtype=tf.float32)
activations = tf.ones([batch, atoms, w, h, depth,1 ,1], dtype=tf.float32)


high_poses, high_activations = r.fit(votes, activations)
print("kernel")
print("got " + str(high_poses.shape))
print("should have been " + str([batch, w, h, atoms] + representation_dim))

print("got " + str(high_activations.shape))
print("should have been " + str([batch, w, h, atoms]))