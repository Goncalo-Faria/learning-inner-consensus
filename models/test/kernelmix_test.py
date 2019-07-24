import tensorflow as tf

from models.coreimp.commonKernels import DotProd, Poly, SpectralMixture, GaussianKernel
from models.coreimp.kernelmix import MonoKernelMix, KernelMix

tf.compat.v1.enable_eager_execution()

dp_test = DotProd()
poly_test = Poly(4)
sm_test = SpectralMixture()
rbf_test = GaussianKernel()

test_f = KernelMix(kernel_list=[dp_test, poly_test, sm_test, rbf_test])
test_m = MonoKernelMix(kernel= sm_test, degree=10)

batch = 1
w = 1
h = 1
depth = 1
representation_dim = [2, 2]

##  input_tensor == {  batch, w , h , depth } + repdim , {batch, w, h,  depth }
test_tensor = tf.ones([batch, w, h, depth] + representation_dim, dtype=tf.float32)

test_result = test_f.apply(test_tensor, test_tensor)

print(test_result.shape)
print(test_result)


test_result = test_m.apply(test_tensor, test_tensor)

print(test_result.shape)
print(test_result)