import tensorflow as tf

from coreimp.commonMetrics import SquaredFrobenius

tf.compat.v1.enable_eager_execution()

norm = SquaredFrobenius()

batch = 1
w = 1
h = 1
depth = 1
representation_dim = [2, 2]

##  input_tensor == {  batch, w , h , depth } + repdim , {batch, w, h,  depth }
test_tensor = tf.ones([batch, w, h, depth] + representation_dim, dtype=tf.float32)

test_result = norm.apply(test_tensor)

print("got " + str(test_result))
print("should have been " + str(4))
