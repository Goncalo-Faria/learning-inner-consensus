import tensorflow as tf

from models.coreimp.commonKernels import DotProd
from models.coreimp.commonMetrics import Frobenius
from models.coreimp.kernelRouting import KernelRouting
from models.coreimp.ninRouting import NiNRouting
from models.coreimp.rnnRouting import RNNRouting

tf.compat.v1.enable_eager_execution()

batch = 32
w = 6
h = 3
depth = 3
representation_dim = [2, 2]
atoms = 4

print("batch:" + str(batch))
print("w:" + str(w))
print("h:" + str(h))
print("depth:" + str(depth))
print("repdim:" + str(representation_dim))
print("out:" + str(atoms) )

r = KernelRouting(
    DotProd(),
    Frobenius(),
    iterations=3
)

r2 = NiNRouting(
    metric=Frobenius(),
    iterations=3,
    activation_layers=[32,10],
    compatibility_layers=[32,10],
    degree=16
)

r3 = RNNRouting(
    metric=Frobenius(),
    iterations=3,
    degree=16
)

votes = tf.ones([batch, atoms, w, h, depth] + representation_dim, dtype=tf.float32)
activations = tf.ones([batch, atoms, w, h, depth], dtype=tf.float32)

high_poses, high_activations = r.fit(votes, activations)
print("kernel")
print("got " + str(high_poses.shape))
print("should have been " + str([batch, w, h, atoms] + representation_dim))

print("got " + str(high_activations.shape))
print("should have been " + str([batch, w, h, atoms]))

high_poses, high_activations = r2.fit(votes, activations)

print("nin")
print("got " + str(high_poses.shape))
print("should have been " + str([batch, w, h, atoms] + representation_dim))

print("got " + str(high_activations.shape))
print("should have been " + str([batch, w, h, atoms]))

#print(high_activations)
#print(high_poses)

high_poses, high_activations = r3.fit(votes, activations)

print("rnn")
print("got " + str(high_poses.shape))
print("should have been " + str([batch, w, h, atoms] + representation_dim))

print("got " + str(high_activations.shape))
print("should have been " + str([batch, w, h, atoms]))