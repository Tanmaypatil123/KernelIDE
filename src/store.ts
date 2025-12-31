import type { Settings, Language } from './types';

const STORAGE_KEY = 'kernelide_settings';
const ENDPOINT_KEY = 'kernelide_endpoint';

export const defaultSettings: Settings = {
  apiKey: '',
  apiSecret: '',
  gpu: 'T4',
  timeout: 30,
};

export function loadSettings(): Settings {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      return { ...defaultSettings, ...JSON.parse(stored) };
    }
  } catch (e) {
    console.error('Failed to load settings:', e);
  }
  return defaultSettings;
}

export function saveSettings(settings: Settings): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  } catch (e) {
    console.error('Failed to save settings:', e);
  }
}

export function getStoredEndpoint(): string | null {
  return localStorage.getItem(ENDPOINT_KEY);
}

export function setStoredEndpoint(endpoint: string): void {
  localStorage.setItem(ENDPOINT_KEY, endpoint);
}

export function clearStoredEndpoint(): void {
  localStorage.removeItem(ENDPOINT_KEY);
}

export const defaultCode: Record<Language, string> = {
  cuda: `#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 1024;
    size_t size = N * sizeof(float);
    
    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    
    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // Copy to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    
    // Copy result back
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // Verify
    printf("Vector addition completed!\\n");
    printf("Result[0] = %f (expected: %f)\\n", h_c[0], h_a[0] + h_b[0]);
    printf("Result[100] = %f (expected: %f)\\n", h_c[100], h_a[100] + h_b[100]);
    
    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    
    return 0;
}`,

  triton: `import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    return output

# Test the kernel
size = 98432
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')

output_triton = add(x, y)
output_torch = x + y

print(f"Triton kernel test:")
print(f"Input size: {size}")
print(f"Max difference: {torch.max(torch.abs(output_triton - output_torch))}")
print(f"Results match: {torch.allclose(output_triton, output_torch)}")
`,

  mojo: `# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

# DOC: mojo/docs/manual/gpu/intro-tutorial.mdx

from math import ceildiv
from sys import has_accelerator

from gpu.host import DeviceContext
from gpu import block_dim, block_idx, thread_idx
from layout import Layout, LayoutTensor

# Vector data type and size
comptime float_dtype = DType.float32
comptime vector_size = 1000
comptime layout = Layout.row_major(vector_size)

# Calculate the number of thread blocks needed by dividing the vector size
# by the block size and rounding up.
comptime block_size = 256
comptime num_blocks = ceildiv(vector_size, block_size)


fn vector_addition(
    lhs_tensor: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    rhs_tensor: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    out_tensor: LayoutTensor[float_dtype, layout, MutAnyOrigin],
):
    """Calculate the element-wise sum of two vectors on the GPU."""

    # Calculate the index of the vector element for the thread to process
    var tid = block_idx.x * block_dim.x + thread_idx.x

    # Don't process out of bounds elements
    if tid < vector_size:
        out_tensor[tid] = lhs_tensor[tid] + rhs_tensor[tid]


def main():
    @parameter
    if not has_accelerator():
        print("No compatible GPU found")
    else:
        # Get the context for the attached GPU
        ctx = DeviceContext()

        # Create HostBuffers for input vectors
        lhs_host_buffer = ctx.enqueue_create_host_buffer[float_dtype](
            vector_size
        )
        rhs_host_buffer = ctx.enqueue_create_host_buffer[float_dtype](
            vector_size
        )
        ctx.synchronize()

        # Initialize the input vectors
        for i in range(vector_size):
            lhs_host_buffer[i] = Float32(i)
            rhs_host_buffer[i] = Float32(i * 0.5)

        print("LHS buffer: ", lhs_host_buffer)
        print("RHS buffer: ", rhs_host_buffer)

        # Create DeviceBuffers for the input vectors
        lhs_device_buffer = ctx.enqueue_create_buffer[float_dtype](vector_size)
        rhs_device_buffer = ctx.enqueue_create_buffer[float_dtype](vector_size)

        # Copy the input vectors from the HostBuffers to the DeviceBuffers
        ctx.enqueue_copy(dst_buf=lhs_device_buffer, src_buf=lhs_host_buffer)
        ctx.enqueue_copy(dst_buf=rhs_device_buffer, src_buf=rhs_host_buffer)

        # Create a DeviceBuffer for the result vector
        result_device_buffer = ctx.enqueue_create_buffer[float_dtype](
            vector_size
        )

        # Wrap the DeviceBuffers in LayoutTensors
        lhs_tensor = LayoutTensor[float_dtype, layout](lhs_device_buffer)
        rhs_tensor = LayoutTensor[float_dtype, layout](rhs_device_buffer)
        result_tensor = LayoutTensor[float_dtype, layout](result_device_buffer)

        # Compile and enqueue the kernel
        ctx.enqueue_function_checked[vector_addition, vector_addition](
            lhs_tensor,
            rhs_tensor,
            result_tensor,
            grid_dim=num_blocks,
            block_dim=block_size,
        )

        # Create a HostBuffer for the result vector
        result_host_buffer = ctx.enqueue_create_host_buffer[float_dtype](
            vector_size
        )

        # Copy the result vector from the DeviceBuffer to the HostBuffer
        ctx.enqueue_copy(
            dst_buf=result_host_buffer, src_buf=result_device_buffer
        )

        # Finally, synchronize the DeviceContext to run all enqueued operations
        ctx.synchronize()

        print("Result vector:", result_host_buffer)`,

  cutlass: `#include <iostream>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/device/gemm.h>
#include <cutlass/util/reference/host/tensor_compare.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/tensor_view_io.h>

// Define GEMM configuration using CUTLASS
using ColumnMajor = cutlass::layout::ColumnMajor;
using RowMajor = cutlass::layout::RowMajor;

using CutlassGemm = cutlass::gemm::device::Gemm<
    float,        // Element type for A matrix
    ColumnMajor,  // Layout for A matrix
    float,        // Element type for B matrix
    RowMajor,     // Layout for B matrix
    float,        // Element type for C matrix
    ColumnMajor,  // Layout for C matrix
    float,        // Element type for accumulator
    cutlass::arch::OpClassSimt,  // Operation class
    cutlass::arch::Sm70          // Target architecture
>;

int main() {
    // Problem size
    int M = 512;
    int N = 512;
    int K = 512;
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Allocate tensors
    cutlass::HostTensor<float, ColumnMajor> A({M, K});
    cutlass::HostTensor<float, RowMajor> B({K, N});
    cutlass::HostTensor<float, ColumnMajor> C({M, N});
    cutlass::HostTensor<float, ColumnMajor> D({M, N});
    
    // Initialize with random data
    cutlass::reference::host::TensorFillRandomUniform(A.host_view(), 1, -1.0f, 1.0f, 0);
    cutlass::reference::host::TensorFillRandomUniform(B.host_view(), 2, -1.0f, 1.0f, 0);
    cutlass::reference::host::TensorFill(C.host_view());
    cutlass::reference::host::TensorFill(D.host_view());
    
    // Copy to device
    A.sync_device();
    B.sync_device();
    C.sync_device();
    D.sync_device();
    
    // Create GEMM operator
    CutlassGemm gemm_op;
    
    // Configure GEMM arguments
    CutlassGemm::Arguments args(
        {M, N, K},
        {A.device_data(), A.stride(0)},
        {B.device_data(), B.stride(0)},
        {C.device_data(), C.stride(0)},
        {D.device_data(), D.stride(0)},
        {alpha, beta}
    );
    
    // Launch GEMM
    cutlass::Status status = gemm_op(args);
    
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM failed!" << std::endl;
        return -1;
    }
    
    cudaDeviceSynchronize();
    
    // Copy result back
    D.sync_host();
    
    std::cout << "CUTLASS GEMM completed successfully!" << std::endl;
    std::cout << "Matrix dimensions: " << M << " x " << N << " x " << K << std::endl;
    std::cout << "D[0,0] = " << D.at({0, 0}) << std::endl;
    
    return 0;
}`,

  cutedsl: `
import cutlass
import cutlass.cute as cute

@cute.kernel
def kernel():
    # Get the x component of the thread index (y and z components are unused)
    tidx, _, _ = cute.arch.thread_idx()
    # Only the first thread (thread 0) prints the message
    if tidx == 0:
        cute.printf("Hello world")
  
@cute.jit
def hello_world():
    # Print hello world from host code
    cute.printf("hello world")

    # Launch kernel
    kernel().launch(
        grid=(1, 1, 1),  # Single thread block
        block=(32, 1, 1),  # One warp (32 threads) per thread block
    )

# Initialize CUDA context for launching a kernel with error checking
# We make context initialization explicit to allow users to control the context creation
# and avoid potential issues with multiple contexts
cutlass.cuda.initialize_cuda_context()

# Method 1: Just-In-Time (JIT) compilation - compiles and runs the code immediately
print("Running hello_world()...")
hello_world()

# Method 2: Compile first (useful if you want to run the same code multiple times)
print("Compiling...")
hello_world_compiled = cute.compile(hello_world)

# Dump PTX/CUBIN files while compiling
from cutlass.cute import KeepPTX, KeepCUBIN

print("Compiling with PTX/CUBIN dumped...")
# Alternatively, compile with string based options like
# cute.compile(hello_world, options="--keep-ptx --keep-cubin") would also work.
hello_world_compiled_ptx_on = cute.compile[KeepPTX, KeepCUBIN](hello_world)

# Run the pre-compiled version
print("Running compiled version...")
hello_world_compiled()

    

`,
};
