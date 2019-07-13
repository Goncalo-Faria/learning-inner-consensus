import tensorflow as tf

from coreimp.commonKernels import DotProd, Poly

tf.compat.v1.enable_eager_execution()

dp_test = DotProd()
poly_test = Poly(4)

batch = 1
w = 1
h = 1
depth = 1
representation_dim = [2, 2]

##  input_tensor == {  batch, w , h , depth } + repdim , {batch, w, h,  depth }
test_tensor = tf.ones([batch, w, h, depth] + representation_dim, dtype=tf.float32)

test_result = dp_test.apply(test_tensor, test_tensor)

print("got " + str(test_result))
print("should have been " + str(4))

test_result = poly_test.apply(test_tensor, test_tensor)

print("got " + str(test_result))
print("should have been " + str(5 ** 4))
