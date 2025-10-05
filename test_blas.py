import numpy as np
import pytart
import scipy.linalg.blas as blas
import os

tart = pytart.Instance()
dev = tart.create_device(0)

def compile_shader(fn):
	tmp = "tmp.spv"
	if os.path.exists(tmp):
		os.remove(tmp)
	os.system(f"glslc -fshader-stage=compute {fn} -o {tmp}")
	with open(tmp, "rb") as f:
		spv = f.read()
	if os.path.exists(tmp):
		os.remove(tmp)
	return spv

sum_module = dev.load_shader(compile_shader("sum.glsl") )
sum_pipeline = dev.create_pipeline(sum_module, "main", push_constants = np.array([1, 1], dtype = np.uint32))

def invoke_sum(size, a_buf, a_stride, out_buf):
	sum_pipeline.dispatch([1, 1, 1], [a_buf, out_buf], np.array([size, a_stride], dtype = np.uint32))
	dev.sync()
	
def test_sum():
	a = np.random.randn(32).astype(np.float32)
	a_buf = dev.allocate_buffer(a.nbytes)
	a_buf.copy_in(a)
	
	b_buf = dev.allocate_buffer(4)
	
	invoke_sum(32, a_buf, 1, b_buf)
	b = np.zeros(1, dtype = np.float32)
	b_buf.copy_out(b)
	assert np.allclose(b, np.sum(a))

dot_module = dev.load_shader(compile_shader("dot.glsl") )
dot_fixed_pipeline = dev.create_pipeline(dot_module, "main", push_constants = np.zeros(3, dtype = np.uint32) )

def invoke_dot(size, x_buf, incx, y_buf, incy, out_buf):
	dot_fixed_pipeline.dispatch([1, 1, 1], [x_buf, y_buf, out_buf], np.array([size, incx, incy], dtype = np.uint32))
	dev.sync()

def test_sdot():
	SIZE = 1024
	a = np.arange(SIZE).astype(np.float32)
	b = np.linspace(0.0, 32.0, num = SIZE).astype(np.float32)
	a_buf = dev.allocate_buffer(a.nbytes)
	b_buf = dev.allocate_buffer(b.nbytes)
	a_buf.copy_in(a)
	b_buf.copy_in(b)
	c_buf = dev.allocate_buffer(4)
	
	invoke_dot(SIZE, a_buf, 1, b_buf, 1, c_buf)
	
	# now we read
	c = np.zeros(1, dtype = np.float32)
	c_buf.copy_out(c)
	assert np.allclose(np.dot(a, b), c)
	
	dev.deallocate_buffer(a_buf)
	dev.deallocate_buffer(b_buf)
	dev.deallocate_buffer(c_buf)
	
asum_module = dev.load_shader(compile_shader("asum.glsl") )
asum_pipeline = dev.create_pipeline(asum_module, "main", push_constants = np.array([1, 1], dtype = np.uint32))

def invoke_asum(n, x_buf, incx, result_buf):
	sequence= dev.create_sequence()
	sequence.record_pipeline(asum_pipeline, [1, 1, 1], [x_buf, result_buf], np.array([n, incx], dtype = np.uint32) )
	dev.submit_sequence(sequence)
	dev.sync()
	
def test_asum():
	a = np.random.randn(32).astype(np.float32)
	a_buf = dev.allocate_buffer(a.nbytes)
	a_buf.copy_in(a)
	
	b_buf = dev.allocate_buffer(4)
	
	invoke_asum(32, a_buf, 1, b_buf)
	b = np.zeros(1, dtype = np.float32)
	b_buf.copy_out(b)
	assert np.allclose(b, np.sum(np.abs(a)))

nrm2_module = dev.load_shader(compile_shader("nrm2.glsl") )
nrm2_pipeline = dev.create_pipeline(nrm2_module, "main", push_constants = np.array([1, 1], dtype = np.uint32))

def invoke_nrm2(n, x_buf, incx, result_buf):
	sequence= dev.create_sequence()
	sequence.record_pipeline(nrm2_pipeline, [1, 1, 1], [x_buf, result_buf], np.array([n, incx], dtype = np.uint32) )
	dev.submit_sequence(sequence)
	dev.sync()
	

def test_nrm2():
	SIZE=4097
	a = np.random.randn(SIZE).astype(np.float32)
	a_buf = dev.allocate_buffer(a.nbytes)
	a_buf.copy_in(a)
	
	b_buf = dev.allocate_buffer(4)
	invoke_nrm2(SIZE, a_buf, 1, b_buf)
	
	b_expected = blas.snrm2(a, SIZE, 0, 1)
	b = np.zeros(1, dtype = np.float32)
	b_buf.copy_out(b)
	assert np.allclose(b_expected, b)

amax_module = dev.load_shader(compile_shader("amax.glsl") )
amax_pipeline = dev.create_pipeline(amax_module, "main", push_constants = np.array([1, 1], dtype = np.uint32))

def invoke_amax(n, x_buf, incx, result_buf):
	sequence= dev.create_sequence()
	sequence.record_pipeline(amax_pipeline, [1, 1, 1], [x_buf, result_buf], np.array([n, incx], dtype = np.uint32) )
	dev.submit_sequence(sequence)
	dev.sync()

def test_amax():
	SIZE=4097
	a = np.random.randn(SIZE).astype(np.float32)
	a_buf = dev.allocate_buffer(a.nbytes)
	a_buf.copy_in(a)
	
	b_buf = dev.allocate_buffer(4)
	invoke_amax(SIZE, a_buf, 1, b_buf)
	
	b_expected = blas.isamax(a, SIZE, 0, 1)
	b = np.zeros(1, dtype = np.uint32)
	b_buf.copy_out(b)
	assert np.allclose(b_expected, b)

amin_module = dev.load_shader(compile_shader("amin.glsl") )
amin_pipeline = dev.create_pipeline(amin_module, "main", push_constants = np.array([1, 1], dtype = np.uint32))

def invoke_amin(n, x_buf, incx, result_buf):
	sequence= dev.create_sequence()
	sequence.record_pipeline(amin_pipeline, [1, 1, 1], [x_buf, result_buf], np.array([n, incx], dtype = np.uint32) )
	dev.submit_sequence(sequence)
	dev.sync()

def test_amin():
	SIZE=4097
	a = np.random.randn(SIZE).astype(np.float32)
	a_buf = dev.allocate_buffer(a.nbytes)
	a_buf.copy_in(a)
	
	b_buf = dev.allocate_buffer(4)
	invoke_amin(SIZE, a_buf, 1, b_buf)
	
	b_expected = np.argmin(np.abs(a))
	b = np.zeros(1, dtype = np.uint32)
	b_buf.copy_out(b)
	assert np.allclose(b_expected, b)

max_module = dev.load_shader(compile_shader("max.glsl") )
max_pipeline = dev.create_pipeline(max_module, "main", push_constants = np.array([1, 1], dtype = np.uint32))

def invoke_max(n, x_buf, incx, result_buf):
	sequence= dev.create_sequence()
	sequence.record_pipeline(max_pipeline, [1, 1, 1], [x_buf, result_buf], np.array([n, incx], dtype = np.uint32) )
	dev.submit_sequence(sequence)
	dev.sync()

def test_max():
	SIZE=4097
	a = np.random.randn(SIZE).astype(np.float32)
	a_buf = dev.allocate_buffer(a.nbytes)
	a_buf.copy_in(a)
	
	b_buf = dev.allocate_buffer(4)
	invoke_max(SIZE, a_buf, 1, b_buf)
	
	b_expected = np.argmax(a)
	b = np.zeros(1, dtype = np.uint32)
	b_buf.copy_out(b)
	assert np.allclose(b_expected, b)

min_module = dev.load_shader(compile_shader("min.glsl") )
min_pipeline = dev.create_pipeline(min_module, "main", push_constants = np.array([1, 1], dtype = np.uint32))

def invoke_min(n, x_buf, incx, result_buf):
	sequence= dev.create_sequence()
	sequence.record_pipeline(min_pipeline, [1, 1, 1], [x_buf, result_buf], np.array([n, incx], dtype = np.uint32) )
	dev.submit_sequence(sequence)
	dev.sync()

def test_min():
	SIZE=4097
	a = np.random.randn(SIZE).astype(np.float32)
	a_buf = dev.allocate_buffer(a.nbytes)
	a_buf.copy_in(a)
	
	b_buf = dev.allocate_buffer(4)
	invoke_min(SIZE, a_buf, 1, b_buf)
	
	b_expected = np.argmin(a)
	b = np.zeros(1, dtype = np.uint32)
	b_buf.copy_out(b)
	assert np.allclose(b_expected, b)

axpy_module = dev.load_shader(compile_shader("axpy.glsl") )
axpy_pipeline = dev.create_pipeline(axpy_module, "main", push_constants = np.zeros(4, dtype = np.uint32) )

def invoke_axpy(n, alpha, x_buf, incx, y_buf, incy):
	# first pack the push constants
	consts = np.zeros(4, dtype = np.uint32)
	consts.view(np.int32)[0] = n
	consts.view(np.float32)[1] = alpha
	consts.view(np.int32)[2] = incx
	consts.view(np.int32)[3] = incy
	
	# then execute the pipeline
	axpy_pipeline.dispatch([n], [x_buf, y_buf], consts)
	dev.sync()

def test_axpy():
	SIZE = 8
	x = np.arange(SIZE).astype(np.float32)
	y = np.linspace(20.0, 52.0, num = SIZE, dtype = np.float32)
	
	x_buf = dev.allocate_buffer(x.nbytes)
	y_buf = dev.allocate_buffer(y.nbytes)
	
	x_buf.copy_in(x)
	y_buf.copy_in(y)
	
	alpha = 0.45
	
	invoke_axpy(SIZE, alpha, x_buf, -1, y_buf, 1)
	
	z_expected = blas.saxpy(x, y, SIZE, alpha, 0, -1, 0, 1)
	
	z_result = np.zeros_like(y)
	y_buf.copy_out(z_result)
	
	assert np.allclose(z_result, z_expected)


copy_module = dev.load_shader(compile_shader("copy.glsl") )
copy_pipeline = dev.create_pipeline(copy_module, "main", push_constants = np.zeros(3, dtype = np.uint32) )

def invoke_copy(n, x_buf, incx, y_buf, incy):
	# first pack the push constants
	consts = np.array([n, 0, 0], dtype = np.uint32)
	consts.view(np.int32)[1] = incx
	consts.view(np.int32)[2] = incy
	
	# then execute the pipeline
	copy_pipeline.dispatch([n], [x_buf, y_buf], consts)
	dev.sync()

def test_copy():
	SIZE = 2048
	x = np.arange(SIZE).astype(np.float32)
	y = np.linspace(20.0, 52.0, num = SIZE, dtype = np.float32)
	
	x_buf = dev.allocate_buffer(x.nbytes)
	y_buf = dev.allocate_buffer(y.nbytes)
	
	x_buf.copy_in(x)
	y_buf.copy_in(y)
	
	invoke_copy(SIZE, x_buf, -1, y_buf, 1)
	
	z_expected = np.flip(x)
	
	z_result = np.zeros_like(y)
	y_buf.copy_out(z_result)
	
	assert np.allclose(z_result, z_expected)
	

swap_module = dev.load_shader(compile_shader("swap.glsl") )
swap_pipeline = dev.create_pipeline(swap_module, "main", push_constants = np.zeros(3, dtype = np.uint32) )

def invoke_swap(n, x_buf, incx, y_buf, incy):
	# first pack the push constants
	consts = np.array([n, 0, 0], dtype = np.uint32)
	consts.view(np.int32)[1] = incx
	consts.view(np.int32)[2] = incy
	
	# then execute the pipeline
	swap_pipeline.dispatch([n], [x_buf, y_buf], consts)
	dev.sync()

def test_swap():
	SIZE = 2048
	x = np.arange(SIZE).astype(np.float32)
	y = np.linspace(20.0, 52.0, num = SIZE, dtype = np.float32)
	
	x_buf = dev.allocate_buffer(x.nbytes)
	y_buf = dev.allocate_buffer(y.nbytes)
	
	x_buf.copy_in(x)
	y_buf.copy_in(y)
	
	invoke_swap(SIZE, x_buf, 1, y_buf, 1)
	
	y_result = np.zeros_like(y)
	y_buf.copy_out(y_result)
	
	x_result = np.zeros_like(x)
	x_buf.copy_out(x_result)
	
	assert np.allclose(x_result, y)
	assert np.allclose(y_result, x)
	
rot_module = dev.load_shader(compile_shader("rot.glsl") )
rot_pipeline = dev.create_pipeline(rot_module, "main", push_constants = np.zeros(5, dtype = np.uint32) )

def invoke_rot(n, x_buf, incx, y_buf, incy, c, s):
	# first pack the push constants
	consts = np.array([n, 0, 0, 0, 0], dtype = np.uint32)
	consts.view(np.int32)[1] = incx
	consts.view(np.int32)[2] = incy
	consts.view(np.float32)[3] = c
	consts.view(np.float32)[4] = s
	
	# then execute the pipeline
	rot_pipeline.dispatch([n], [x_buf, y_buf], consts)
	dev.sync()

def test_rot():
	SIZE = 2048
	x = np.arange(SIZE).astype(np.float32)
	y = np.linspace(20.0, 52.0, num = SIZE, dtype = np.float32)
	
	x_buf = dev.allocate_buffer(x.nbytes)
	y_buf = dev.allocate_buffer(y.nbytes)
	
	x_buf.copy_in(x)
	y_buf.copy_in(y)
	
	a = np.pi*234
	c = np.cos(a)
	s = np.sin(a)
	
	invoke_rot(SIZE, x_buf, 1, y_buf, 1, c, s)
	
	x_result = np.zeros_like(x)
	x_buf.copy_out(x_result)
	
	y_result = np.zeros_like(y)
	y_buf.copy_out(y_result)
	
	blas.srot(x, y, c, s, SIZE, 0, 1, 0, 1, True, True)
	
	assert np.allclose(x_result, x)
	assert np.allclose(y_result, y)


rotg_module = dev.load_shader(compile_shader("rotg.glsl") )
rotg_pipeline = dev.create_pipeline(rotg_module, "main")

def invoke_rotg(n, a_buf, b_buf, c_buf, s_buf):
	rotg_pipeline.dispatch([n], [a_buf, b_buf, c_buf, s_buf])
	dev.sync()

def test_rotg():
	SIZE = 1
	a = np.array(2.4, dtype = np.float32)
	b = np.array(3.2, dtype = np.float32)
	
	a_buf = dev.allocate_buffer(a.nbytes)
	b_buf = dev.allocate_buffer(b.nbytes)
	
	a_buf.copy_in(a)
	b_buf.copy_in(b)
	
	c_buf = dev.allocate_buffer(a.nbytes)
	s_buf = dev.allocate_buffer(b.nbytes)
	
	invoke_rotg(SIZE, a_buf, b_buf, c_buf, s_buf)
	
	c_expected, s_expected = blas.srotg(a, b)
	c_result = np.array(0.0, dtype = np.float32)
	s_result = np.array(0.0, dtype = np.float32)
	c_buf.copy_out(c_result)
	s_buf.copy_out(s_result)
	
	assert np.allclose(c_result, c_expected)
	assert np.allclose(s_result, s_expected)

rotm_module = dev.load_shader(compile_shader("rotm.glsl") )
rotm_pipeline = dev.create_pipeline(rotm_module, "main", push_constants = np.zeros(8, dtype = np.uint32) )

def invoke_rotm(n, x_buf, incx, y_buf, incy, c, s):
	# first pack the push constants
	consts = np.array([n, 0, 0, 0, 0], dtype = np.uint32)
	consts.view(np.int32)[1] = incx
	consts.view(np.int32)[2] = incy
	consts.view(np.float32)[3] = c
	consts.view(np.float32)[4] = s
	
	# then execute the pipeline
	rotm_pipeline.dispatch([n], [x_buf, y_buf], consts)
	dev.sync()

def test_rotm():
	raise NotImplementedError
	SIZE = 2048
	x = np.arange(SIZE).astype(np.float32)
	y = np.linspace(20.0, 52.0, num = SIZE, dtype = np.float32)
	
	x_buf = dev.allocate_buffer(x.nbytes)
	y_buf = dev.allocate_buffer(y.nbytes)
	
	x_buf.copy_in(x)
	y_buf.copy_in(y)
	
	a = np.pi*234
	c = np.cos(a)
	s = np.sin(a)
	
	invoke_rotm(SIZE, x_buf, 1, y_buf, 1, c, s)
	
	x_result = np.zeros_like(x)
	x_buf.copy_out(x_result)
	
	y_result = np.zeros_like(y)
	y_buf.copy_out(y_result)
	
	blas.srot(x, y, c, s, SIZE, 0, 1, 0, 1, True, True)
	
	assert np.allclose(x_result, x)
	assert np.allclose(y_result, y)
