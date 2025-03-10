import tensorflow as tf

from models.coreimp.commonKernels import DotProd, Poly, SpectralMixture, GaussianKernel

tf.compat.v1.enable_eager_execution()

dp_test = DotProd()
poly_test = Poly(4)
sm_test = SpectralMixture()
rbf_test = GaussianKernel()

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
print(test_result.shape)

test_result = poly_test.apply(test_tensor, test_tensor)

print("got " + str(test_result))
print("should have been " + str(5 ** 4))
print(test_result.shape)

test_result = sm_test.apply(test_tensor, test_tensor)

print(" sm ")
print(test_result.shape)
print(test_result)

test_result = rbf_test.apply(test_tensor, test_tensor)

print(" rbf ")
print(test_result.shape)
print(test_result)

