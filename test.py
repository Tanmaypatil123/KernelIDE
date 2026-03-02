# This examples uses CuPy which can be installed via `pip install cupy-cuda13x`
# Make sure cuda toolkit 13.1+ is installed: https://developer.nvidia.com/cuda-downloads

import cuda.tile as ct
import cupy
import numpy as np

TILE_SIZE = 16

# cuTile kernel for adding two dense vectors. It runs in parallel on the GPU.
@ct.kernel
def vector_add_kernel(a, b, result):
    block_id = ct.bid(0)
    a_tile = ct.load(a, index=(block_id,), shape=(TILE_SIZE,))
    b_tile = ct.load(b, index=(block_id,), shape=(TILE_SIZE,))
    result_tile = a_tile + b_tile
    ct.store(result, index=(block_id,), tile=result_tile)

# Generate input arrays
rng = cupy.random.default_rng()
a = rng.random(128)
b = rng.random(128)
expected = cupy.asnumpy(a) + cupy.asnumpy(b)

# Allocate an output array and launch the kernel
result = cupy.zeros_like(a)
grid = (ct.cdiv(a.shape[0], TILE_SIZE), 1, 1)
ct.launch(cupy.cuda.get_current_stream(), grid, vector_add_kernel, (a, b, result))

# Verify the results
result_np = cupy.asnumpy(result)
np.testing.assert_array_almost_equal(result_np, expected)