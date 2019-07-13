import tensorflow as tf

from models.coreimp.commonMetrics import SquaredFrobenius
from models.coreimp.commonKernels import DotProd
from models.coreimp.kernelRouting import KernelRouting

tf.compat.v1.enable_eager_execution()

batch = 32
w = 6
h = 3
depth = 3
representation_dim = [2, 2]
atoms = 4

r = KernelRouting(
        DotProd(),
        SquaredFrobenius(),
        iterations=3
)

votes = tf.ones([batch, atoms ,w, h, depth] + representation_dim, dtype=tf.float32)
activations = tf.ones([batch, atoms, w, h, depth] , dtype=tf.float32)

high_poses, high_activations = r.fit(votes, activations)

print("got " + str(high_poses.shape))
print("should have been " + str([batch,w,h,atoms] + representation_dim))

print("got " + str(high_activations.shape))
print("should have been " + str([batch,w,h,atoms]))

