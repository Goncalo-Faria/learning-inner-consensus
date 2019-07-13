import tensorflow as tf

from layers.capsule import CapsuleLayer
from coreimp.equiTransform import EquiTransform
from coreimp.kernelRouting import KernelRouting
from coreimp.commonKernels import DotProd
from coreimp.commonMetrics import SquaredFrobenius

tf.compat.v1.enable_eager_execution()

test_layer = CapsuleLayer(
    routing = KernelRouting(
        kernel=DotProd(),
        metric=SquaredFrobenius(),
        iterations=10,
        name="fuzzykmeans",
        verbose=True
        ),
    transform=EquiTransform(
        output_atoms=2,
        metric=SquaredFrobenius(),
        verbose=False
        ),
    ksizes=[1,2,2,1],
    name="unitCapsule"
)

batch = 32
w = 2
h = 2
depth = 3
representation_dim = [4,4]

##  input_tensor == {  batch, w , h , depth } + repdim , {batch, w, h,  depth }
test_tensor = tf.ones( [batch,w,h,depth] + representation_dim, dtype=tf.float32), tf.ones( [batch,w,h,depth], dtype=tf.float32)

pose , activation = test_layer.inference( test_tensor )

print(pose.shape)
print(activation.shape)

