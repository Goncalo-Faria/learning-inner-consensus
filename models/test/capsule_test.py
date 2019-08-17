import tensorflow as tf

from models.coreimp.commonKernels import DotProd
from models.coreimp.commonMetrics import SquaredFrobenius
from models.coreimp.equiTransform import EquiTransform
from models.coreimp.kernelRouting import KernelRouting
from models.layers.capsule import CapsuleLayer, FullyConnectedCapsuleLayer, CapsuleClassLayer, PrimaryCapsuleLayer

tf.compat.v1.enable_eager_execution()

test_primary_layer = PrimaryCapsuleLayer(
    pose_dim=[4, 4],
    ksize=[5, 5],
    groups=12
)

test_conv_layer = CapsuleLayer(
    routing=KernelRouting(
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
    ksizes=[1, 2, 2, 1],
    name="unitCapsule"
)

test_connected_layer = FullyConnectedCapsuleLayer(
    routing=KernelRouting(
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
    name="fullyCapsule"
)

test_final_layer = CapsuleClassLayer(
    normalized=True,
    name="final",
)

batch = 32
w = 8
h = 8
depth = 3
representation_dim = [4, 4]

##  input_tensor == {  batch, w , h , depth } + repdim , {batch, w, h,  depth }
test_tensor = tf.ones([batch, w, h, depth] + representation_dim, dtype=tf.float32), tf.ones([batch, w, h, depth],
                                                                                            dtype=tf.float32)

pose, activation = test_conv_layer.inference(test_tensor)

print("poses: " + str(test_tensor[0].shape))
print("act  : " + str(test_tensor[1].shape))

print("primary")
pri_poses, pri_activations = test_primary_layer.inference(test_tensor[1])
print(pri_poses.shape)
print(pri_activations.shape)

print("convolutional")
print(pose.shape)
print(activation.shape)

pose, activation = test_connected_layer.inference(test_tensor)

print("connected")
print(pose.shape)
print(activation.shape)

pose, activation = test_final_layer.inference(test_tensor)

print("final")
print(pose.shape)
print(activation.shape)
