import numpy as np
import pytart
import scipy.linalg.blas as blas
import os
import pytest
import enum
import copy

tart = pytart.Instance()
dev = tart.create_device(0)

class EOrder(enum.Enum):
	ROW_MAJOR = 101
	COLUMN_MAJOR = 102

class ETranspose(enum.Enum):
	NO_TRANSPOSE = 111
	TRANSPOSE = 112
	CONJ_TRANSPOSE = 113
	CONJ_NO_TRANSPOSE = 114

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

def invoke_rotm(n, x_buf, incx, y_buf, incy, param):
	# first pack the push constants
	consts = np.array([n, 0, 0, 0, 0, 0, 0, 0], dtype = np.uint32)
	consts.view(np.int32)[1] = incx
	consts.view(np.int32)[2] = incy
	consts.view(np.float32)[3:8] = param.astype(np.float32)
	
	# then execute the pipeline
	rotm_pipeline.dispatch([n], [x_buf, y_buf], consts)
	dev.sync()

def test_rotm():
	for flag in [-1.0, 0.0, 1.0, -2.0]:
		SIZE = 2048
		x = np.arange(SIZE).astype(np.float32)
		y = np.linspace(20.0, 52.0, num = SIZE, dtype = np.float32)
		
		x_buf = dev.allocate_buffer(x.nbytes)
		y_buf = dev.allocate_buffer(y.nbytes)
		
		x_buf.copy_in(x)
		y_buf.copy_in(y)
		
		param = np.array([0.0, 23.6, -20.5, 2.0, 6.892], dtype = np.float32)
		param[0] = flag
		
		invoke_rotm(SIZE, x_buf, 1, y_buf, 1, param)
		
		x_result = np.zeros_like(x)
		x_buf.copy_out(x_result)
		
		y_result = np.zeros_like(y)
		y_buf.copy_out(y_result)
		
		blas.srotm(x, y, param, SIZE, 0, 1, 0, 1, True, True)
		
		assert np.allclose(x_result, x)
		assert np.allclose(y_result, y)

@pytest.mark.xfail
def test_rotmg():
	# not going to worry about this one for now, given how un-parallelizable it is
	raise NotImplementedError
	

scal_module = dev.load_shader(compile_shader("scal.glsl") )
scal_pipeline = dev.create_pipeline(scal_module, "main", push_constants = np.zeros(3, dtype = np.uint32) )

def invoke_scal(n, alpha, x_buf, incx):
	# first pack the push constants
	consts = np.array([n, 0, 0], dtype = np.uint32)
	consts.view(np.float32)[1] = alpha
	consts.view(np.int32)[2] = incx
	
	# then execute the pipeline
	scal_pipeline.dispatch([n], [x_buf], consts)
	dev.sync()

def test_scal():
	SIZE = 2048
	x = np.arange(SIZE).astype(np.float32)
	x_buf = dev.allocate_buffer(x.nbytes)
	x_buf.copy_in(x)
	
	alpha = 4.5
	
	invoke_scal(SIZE, alpha, x_buf, 1)
	
	x_result = np.zeros_like(x)
	x_buf.copy_out(x_result)
	assert np.allclose(x_result, x*alpha)
	
	
gemv_module = dev.load_shader(compile_shader("gemv.glsl") )
gemv_pipelines = {}
def get_gemv_pipeline(column_size: int):
	if not column_size in gemv_pipelines.keys():
		ls = np.array([column_size], dtype = np.uint32)
		gemv_pipelines[column_size] = dev.create_pipeline(gemv_module, "main", ls, np.zeros(9).astype(np.float32) )
	return gemv_pipelines[column_size]
	
def invoke_gemv(
		order,
		trans: ETranspose,
		m: int, n: int, alpha: float,
		A_buf: pytart.Buffer, lda: int,
		x_buf: pytart.Buffer, incx: int,
		beta: float,
		y_buf: pytart.Buffer, incy: int
	):
	
	# pack push constants
	push = np.zeros(9, dtype = np.float32)
	push.view(np.uint32)[0] = order.value
	push.view(np.uint32)[1] = trans.value
	push.view(np.float32)[2] =  alpha
	push.view(np.uint32)[3] = lda
	push.view(np.int32)[4] = incx
	push.view(np.float32)[5] = beta
	push.view(np.int32)[6] = incy
	push.view(np.uint32)[7] = m
	push.view(np.uint32)[8] = m
	
	# initialize correct pipeline
	gemv_pipeline = get_gemv_pipeline(m)
	
	# dispatch
	gemv_pipeline.dispatch([n, 1, 1], [x_buf, A_buf, y_buf], push)
	dev.sync()

def reference_gemv(
		order: EOrder,
		trans: ETranspose,
		m: int, n: int, alpha: float,
		A_buf: np.ndarray, lda: int,
		x_buf: np.ndarray, incx: int,
		beta: float,
		y_buf: np.ndarray, incy: int
	):
	# column major basically just means transposed by default.
	# I am doing it internally here just so I don't lose my mind trying to keep track of multiple transposes
	if trans == ETranspose.NO_TRANSPOSE:
		if order == EOrder.ROW_MAJOR:
			A_buf = A_buf.T
		y_buf[:] = blas.sgemv(alpha, A_buf, x_buf, beta, y_buf)
	else:
		# lda should be the column size if not transposed, which with scipy's weird behavior will be the second shape element
		if order == EOrder.COLUMN_MAJOR:
			A_buf = A_buf.T
		y_buf[:] = blas.sgemv(alpha, A_buf, x_buf, beta, y_buf)
	


def test_gemv_row_major():
	# just do row major first, since python, C++, and literally everything else uses this natively
	order = EOrder.ROW_MAJOR
	for transpose in [ETranspose.NO_TRANSPOSE, ETranspose.TRANSPOSE]:
		x = np.linspace(5.4, 20.1, num = 4).astype(np.float32)
		a_shape = (4, 8) if transpose == ETranspose.NO_TRANSPOSE else (8, 4)
		A = np.linspace(0.0, 50.7, num = np.prod(a_shape)).reshape(*a_shape).astype(np.float32)
		y = np.linspace(49.2, 100.78, num = 8).astype(np.float32)
		alpha = 5.9
		#beta = 5.48
		beta = 2.4
		
		m = x.shape[0]
		n = y.shape[0]
		
		# lda is whatever dimension that happens to be contiguous. If row-major, it is the row.
		# If column-major, it is the column.
		lda = n if transpose == ETranspose.NO_TRANSPOSE else m
		
		# here we goes
		x_buf = dev.allocate_buffer(x.nbytes)
		A_buf = dev.allocate_buffer(A.nbytes)
		y_buf = dev.allocate_buffer(y.nbytes)
		x_buf.copy_in(x)
		A_buf.copy_in(A)
		y_buf.copy_in(y)
		invoke_gemv(order, transpose, m, n, alpha, A_buf, lda, x_buf, 1, beta, y_buf, 1)
		
		y_expected = np.zeros_like(y)
		y_expected[:] = y
		reference_gemv(order, transpose, m, n, alpha, A, lda, x, 1, beta, y_expected, 1)
		#if not transpose == ETranspose.TRANSPOSE:
		#	y_expected = alpha*np.dot(x, A) + beta*y
		#else:
		#	y_expected = alpha*np.dot(x, A.T) + beta*y
		y_result = np.zeros_like(y)
		y_buf.copy_out(y_result)
		
		assert np.allclose(y_result, y_expected), f"failed on transpose type: {transpose}"

def test_gemv_column_major():
	# just do row major first, since python, C++, and literally everything else uses this natively
	order = EOrder.COLUMN_MAJOR
	for transpose in [ETranspose.NO_TRANSPOSE, ETranspose.TRANSPOSE]:
		x = np.linspace(5.4, 20.1, num = 4).astype(np.float32)
		a_shape = (8, 4) if transpose == ETranspose.NO_TRANSPOSE else (4, 8)
		A = np.linspace(0.0, 50.7, num = np.prod(a_shape)).reshape(*a_shape).astype(np.float32)
		y = np.linspace(49.2, 100.78, num = 8).astype(np.float32)
		alpha = 5.9
		#beta = 5.48
		beta = 2.4
		
		m = x.shape[0]
		n = y.shape[0]
		
		# lda is whatever dimension that happens to be contiguous. If row-major, it is the row.
		# If column-major, it is the column.
		lda = m if transpose == ETranspose.NO_TRANSPOSE else n
		
		# here we goes
		x_buf = dev.allocate_buffer(x.nbytes)
		A_buf = dev.allocate_buffer(A.nbytes)
		y_buf = dev.allocate_buffer(y.nbytes)
		x_buf.copy_in(x)
		A_buf.copy_in(A)
		y_buf.copy_in(y)
		invoke_gemv(order, transpose, m, n, alpha, A_buf, lda, x_buf, 1, beta, y_buf, 1)
		
		y_expected = np.zeros_like(y)
		y_expected[:] = y
		reference_gemv(order, transpose, m, n, alpha, A, lda, x, 1, beta, y_expected, 1)
		if not transpose == ETranspose.TRANSPOSE:
			y_expected = alpha*np.dot(x, A.T) + beta*y
		else:
			y_expected = alpha*np.dot(x, A) + beta*y
		y_result = np.zeros_like(y)
		y_buf.copy_out(y_result)
		
		assert np.allclose(y_result, y_expected), f"failed on transpose type: {transpose}"

ger_module = dev.load_shader(compile_shader("ger.glsl") )
ger_pipeline = dev.create_pipeline(ger_module, "main", push_constants = np.zeros(8, dtype = np.uint32) )

def invoke_ger(order, transpose, m, n, alpha, x_buf, incx, y_buf, incy, A_buf, lda):
	# pack the push constants
	push = np.array([order.value, transpose.value, m, n, 0, incx, incy, lda], dtype = np.uint32)
	push.view(np.float32)[4] = alpha
	ger_pipeline.dispatch([n, m, 1], [x_buf, y_buf, A_buf], push)
	dev.sync()
	

def ger_reference(order, transpose, m, n, alpha, x_buf, incx, y_buf, incy, A_buf, lda):
	if transpose == ETranspose.TRANSPOSE:
		A_buf = A_buf.T
	A_buf[:] = blas.sger(alpha, x_buf, y_buf, incx, incy, A_buf)

# this is also geru, etc.
# still need to implement complex transposing :c
def test_ger():
	#raise NotImplementedError
	for order in [EOrder.ROW_MAJOR, EOrder.COLUMN_MAJOR]:
		for transpose in [ETranspose.NO_TRANSPOSE, ETranspose.TRANSPOSE]:
			x = np.arange(4).astype(np.float32)
			y = np.arange(8).astype(np.float32)
			A = np.linspace(0.0, 1.0, num = 4*8, dtype = np.float32).reshape(4, 8)
			if transpose == ETranspose.TRANSPOSE:
				# modify the test to use a transposed matrix
				A = A.reshape(8, 4)
			
			alpha = 0.5
			m = x.size
			n = y.size
			# lda is the size of the contiguous dimension of A. Which in row major order is the row size, which is at index 1
			lda = A.shape[1]
			
			# compute with shader
			x_buf = dev.allocate_buffer(x.nbytes)
			y_buf = dev.allocate_buffer(y.nbytes)
			A_buf = dev.allocate_buffer(A.nbytes)
			x_buf.copy_in(x)
			y_buf.copy_in(y)
			A_buf.copy_in(A)
			invoke_ger(order, transpose, m, n, alpha, x_buf, 1, y_buf, 1, A_buf, lda)
			A_result = np.zeros_like(A)
			A_buf.copy_out(A_result)
			
			# compute output with reference implementation
			A_expected = np.zeros_like(A)
			A_expected[:] = A
			ger_reference(order, transpose, m, n, alpha, x, 1, y, 1, A_expected, lda)
			
			assert np.allclose(A_expected, A_result), f"failed with order: {order}, transpose: {transpose}"

trsv_module = dev.load_shader(compile_shader("trsv-lower.glsl") )
trsv_pipelines = {}
def get_trsv_pipeline(mat_size: int):
	if not mat_size in trsv_pipelines.keys():
		ls = np.array([mat_size], dtype = np.uint32)
		gemv_pipelines[mat_size] = dev.create_pipeline(trsv_module, "main", ls, np.zeros(4).astype(np.float32) )
	return gemv_pipelines[mat_size]

def invoke_trsv(order, lower, transpose, unit_diag: bool, n: int, A_buf, lda, x_buf, incx):
	consts = np.array([order.value, transpose.value, 0, lda], dtype = np.uint32)
	consts.view(np.int32)[2] = incx	
	pipeline = get_trsv_pipeline(n)
	pipeline.dispatch([n, 1, 1], [x_buf, A_buf], consts)
	dev.sync()

def trsv_reference(order, lower, transpose, unit_diag: bool, n: int, A_buf, lda, x_buf, incx):
	b_copy = x_buf
	if lower:
		L = A_buf
		shared_mem = np.zeros(4, dtype = np.float32)
		for i in range(n):
			# ok, this should be a lot more parallelizable
			# `i` will be the global work group, `j` will be the local invocation
			for j in range(n):
				if j < i:
					# this will be split across multiple local invocations...
					shared_mem[j] = L[i, j]*x_buf[j]
			# barrier here
			# and then this loop will be at invocation 0
			for j in range(n):
				if j < i:
					b_copy[i] = b_copy[i] - shared_mem[j]
			x_buf[i] = b_copy[i] / L[i, i]
	else:
		U = A_buf
		for j in np.flip(np.arange(0, n) ):
			x_buf[j] = b_copy[j]/U[j, j]
			for i in range(j):
				b_copy[i] = b_copy[i] - U[i, j]*x_buf[j]

	
def test_trsv():
	for lower in [True]:
		# currently implementing from reference: https://courses.grainger.illinois.edu/cs554/fa2015/notes/08_triangular_8up.pdf
		b = np.arange(4).astype(np.float32)
		A = np.random.randn(16).astype(np.float32).reshape(4, 4) + 1
		
		x_expected = np.zeros(4, dtype = np.float32)
		x_expected[:] = b
		trsv_reference(EOrder.ROW_MAJOR, lower, False, False, 4, A, 4, x_expected, 1)
		
		if lower:
			P = np.tril(A)
		else:
			P = np.triu(A)
		#assert np.allclose(np.dot(P, x_expected), b, atol = 1.0e-5)
		
		x_buf = dev.allocate_buffer(b.nbytes)
		x_buf.copy_in(b)
		A_buf = dev.allocate_buffer(A.nbytes)
		A_buf.copy_in(A)
		
		invoke_trsv(EOrder.COLUMN_MAJOR, lower, ETranspose.NO_TRANSPOSE, False, 4, A_buf, 4, x_buf, 1)
		x_result = np.zeros_like(b)
		x_buf.copy_out(x_result)
		dummy = np.zeros_like(A)
		A_buf.copy_out(dummy)
		#input(dummy)
		assert np.allclose(x_expected, x_result, atol = 1.0e-5)
