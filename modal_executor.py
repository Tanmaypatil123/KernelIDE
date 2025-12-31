"""
KernelIDE Modal Executor
========================

This is the Modal backend that executes GPU kernels.
Deploy this to your Modal account with:

    modal deploy modal_executor.py

After deployment, copy the endpoint URL and paste it in KernelIDE settings.
The URL format is: https://<your-workspace>--kernelide-executor-api.modal.run
"""

import modal
import subprocess
import tempfile
import os
import time
import sys

app = modal.App("kernelide-executor")

# Base image with CUDA 12.9 toolkit, PyTorch, and Triton
cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu24.04", add_python="3.12")
    .apt_install("build-essential", "git", "wget", "cmake", "ninja-build")
    .pip_install(
        "torch>=2.5.0",
        "triton>=3.1.0",
        "numpy>=2.0.0",
        "pydantic>=2.0.0",
        "cuda-python==12.9.0",
        "nvidia-cutlass-dsl",
    )
)

# Image with CUTLASS 3.7 (latest) for CUTE DSL C++
cutlass_image = (
    cuda_image
    .run_commands(
        "git clone --depth 1 --branch v3.7.0 https://github.com/NVIDIA/cutlass.git /opt/cutlass",
    )
    .env({"CUTLASS_PATH": "/opt/cutlass"})
)

# Image with Mojo SDK
mojo_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu24.04", add_python="3.12")
    .apt_install("build-essential", "git", "wget", "curl")
    .pip_install(
        "mojo",
        extra_index_url="https://modular.gateway.scarf.sh/simple/",
    )
)

# FastAPI image for the web endpoint
web_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi[standard]",
    "pydantic",
)


def get_gpu_config(gpu_type: str):
    """Map GPU type string to Modal GPU config."""
    gpu_map = {
        "T4": "T4",
        "L4": "L4",
        "A10": "A10",
        "A100-40GB": "A100",
        "A100-80GB": "A100-80GB",
        "L40S": "L40S",
        "H100": "H100",
        "H200": "H200",
        "B200": "B200",
    }
    return gpu_map.get(gpu_type, "T4")


@app.function(
    image=cuda_image,
    gpu="T4",
    timeout=300,
    memory=8192,
)
def execute_cuda_t4(code: str, timeout_seconds: int = 30):
    return _execute_cuda(code, timeout_seconds)


@app.function(
    image=cuda_image,
    gpu="A100",
    timeout=300,
    memory=16384,
)
def execute_cuda_a100(code: str, timeout_seconds: int = 30):
    return _execute_cuda(code, timeout_seconds)


@app.function(
    image=cuda_image,
    gpu="H100",
    timeout=300,
    memory=32768,
)
def execute_cuda_h100(code: str, timeout_seconds: int = 30):
    return _execute_cuda(code, timeout_seconds)


@app.function(
    image=cuda_image,
    gpu="B200",
    timeout=300,
    memory=65536,
)
def execute_cuda_b200(code: str, timeout_seconds: int = 30):
    return _execute_cuda(code, timeout_seconds)


def _execute_cuda(code: str, timeout_seconds: int = 30):
    """Execute CUDA C++ code."""
    start_time = time.time()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        source_file = os.path.join(tmpdir, "kernel.cu")
        binary_file = os.path.join(tmpdir, "kernel")
        
        with open(source_file, "w") as f:
            f.write(code)
        
        # Compile with nvcc
        compile_result = subprocess.run(
            ["nvcc", "-o", binary_file, source_file, "-lcudart", "-O2"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        if compile_result.returncode != 0:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Compilation failed:\n{compile_result.stderr}",
                "execution_time": time.time() - start_time,
            }
        
        # Execute the binary
        try:
            run_result = subprocess.run(
                [binary_file],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            return {
                "success": run_result.returncode == 0,
                "stdout": run_result.stdout,
                "stderr": run_result.stderr,
                "execution_time": time.time() - start_time,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution timed out after {timeout_seconds} seconds",
                "execution_time": timeout_seconds,
            }


@app.function(
    image=cuda_image,
    gpu="T4",
    timeout=300,
    memory=8192,
)
def execute_triton_t4(code: str, timeout_seconds: int = 30):
    return _execute_triton(code, timeout_seconds)


@app.function(
    image=cuda_image,
    gpu="A100",
    timeout=300,
    memory=16384,
)
def execute_triton_a100(code: str, timeout_seconds: int = 30):
    return _execute_triton(code, timeout_seconds)


@app.function(
    image=cuda_image,
    gpu="H100",
    timeout=300,
    memory=32768,
)
def execute_triton_h100(code: str, timeout_seconds: int = 30):
    return _execute_triton(code, timeout_seconds)


@app.function(
    image=cuda_image,
    gpu="B200",
    timeout=300,
    memory=65536,
)
def execute_triton_b200(code: str, timeout_seconds: int = 30):
    return _execute_triton(code, timeout_seconds)


def _execute_triton(code: str, timeout_seconds: int = 30):
    """Execute Triton Python code."""
    start_time = time.time()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        source_file = os.path.join(tmpdir, "kernel.py")
        
        with open(source_file, "w") as f:
            f.write(code)
        
        try:
            result = subprocess.run(
                [sys.executable, source_file],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=tmpdir,
                env={**os.environ, "TRITON_CACHE_DIR": tmpdir},
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_time": time.time() - start_time,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution timed out after {timeout_seconds} seconds",
                "execution_time": timeout_seconds,
            }


@app.function(
    image=cutlass_image,
    gpu="T4",
    timeout=300,
    memory=8192,
)
def execute_cutlass_t4(code: str, timeout_seconds: int = 30):
    return _execute_cutlass(code, timeout_seconds)


@app.function(
    image=cutlass_image,
    gpu="A100",
    timeout=300,
    memory=16384,
)
def execute_cutlass_a100(code: str, timeout_seconds: int = 30):
    return _execute_cutlass(code, timeout_seconds)


@app.function(
    image=cutlass_image,
    gpu="H100",
    timeout=300,
    memory=32768,
)
def execute_cutlass_h100(code: str, timeout_seconds: int = 30):
    return _execute_cutlass(code, timeout_seconds)


@app.function(
    image=cutlass_image,
    gpu="B200",
    timeout=300,
    memory=65536,
)
def execute_cutlass_b200(code: str, timeout_seconds: int = 30):
    return _execute_cutlass(code, timeout_seconds)


def _execute_cutlass(code: str, timeout_seconds: int = 30):
    """Execute CUTLASS/CUTE DSL code."""
    start_time = time.time()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        source_file = os.path.join(tmpdir, "kernel.cu")
        binary_file = os.path.join(tmpdir, "kernel")
        
        with open(source_file, "w") as f:
            f.write(code)
        
        cutlass_path = os.environ.get("CUTLASS_PATH", "/opt/cutlass")
        
        # Compile with CUTLASS includes
        compile_result = subprocess.run(
            [
                "nvcc",
                "-o", binary_file,
                source_file,
                f"-I{cutlass_path}/include",
                f"-I{cutlass_path}/tools/util/include",
                "-lcudart",
                "-std=c++17",
                "--expt-relaxed-constexpr",
                "-O2",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        
        if compile_result.returncode != 0:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Compilation failed:\n{compile_result.stderr}",
                "execution_time": time.time() - start_time,
            }
        
        try:
            run_result = subprocess.run(
                [binary_file],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            return {
                "success": run_result.returncode == 0,
                "stdout": run_result.stdout,
                "stderr": run_result.stderr,
                "execution_time": time.time() - start_time,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution timed out after {timeout_seconds} seconds",
                "execution_time": timeout_seconds,
            }


@app.function(
    image=mojo_image,
    gpu="T4",
    timeout=300,
    memory=8192,
)
def execute_mojo_t4(code: str, timeout_seconds: int = 30):
    return _execute_mojo(code, timeout_seconds)


@app.function(
    image=mojo_image,
    gpu="A100",
    timeout=300,
    memory=16384,
)
def execute_mojo_a100(code: str, timeout_seconds: int = 30):
    return _execute_mojo(code, timeout_seconds)


@app.function(
    image=mojo_image,
    gpu="H100",
    timeout=300,
    memory=32768,
)
def execute_mojo_h100(code: str, timeout_seconds: int = 30):
    return _execute_mojo(code, timeout_seconds)


@app.function(
    image=mojo_image,
    gpu="B200",
    timeout=300,
    memory=65536,
)
def execute_mojo_b200(code: str, timeout_seconds: int = 30):
    return _execute_mojo(code, timeout_seconds)


def _execute_mojo(code: str, timeout_seconds: int = 30):
    """Execute Mojo code."""
    start_time = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        source_file = os.path.join(tmpdir, "kernel.mojo")

        with open(source_file, "w") as f:
            f.write(code)

        try:
            # Execute Mojo code directly (Mojo is JIT compiled)
            result = subprocess.run(
                ["mojo", source_file],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=tmpdir,
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_time": time.time() - start_time,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution timed out after {timeout_seconds} seconds",
                "execution_time": timeout_seconds,
            }


def create_web_app():
    """Create FastAPI web app - imported lazily to avoid import on GPU workers."""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    web_app = FastAPI(title="KernelIDE Executor")

    # Enable CORS for browser access
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class ExecuteRequest(BaseModel):
        code: str
        language: str
        gpu_type: str = "T4"
        timeout: int = 30

    class ExecuteResponse(BaseModel):
        success: bool
        stdout: str
        stderr: str
        execution_time: float

    @web_app.post("/execute", response_model=ExecuteResponse)
    async def execute(request: ExecuteRequest):
        """Execute code on GPU."""
        
        # Select executor based on language and GPU type
        gpu = request.gpu_type.upper()
        lang = request.language.lower()
        
        # Map to appropriate executor function
        # We have T4, A100, H100, B200 variants for each language
        # For other GPUs, fall back to closest match
        gpu_tier = "t4"  # default
        if gpu in ["A100-40GB", "A100-80GB", "A100", "A10", "L4", "L40S"]:
            gpu_tier = "a100"
        elif gpu in ["H100", "H200"]:
            gpu_tier = "h100"
        elif gpu == "B200":
            gpu_tier = "b200"
        
        executors = {
            ("cuda", "t4"): execute_cuda_t4,
            ("cuda", "a100"): execute_cuda_a100,
            ("cuda", "h100"): execute_cuda_h100,
            ("cuda", "b200"): execute_cuda_b200,
            ("triton", "t4"): execute_triton_t4,
            ("triton", "a100"): execute_triton_a100,
            ("triton", "h100"): execute_triton_h100,
            ("triton", "b200"): execute_triton_b200,
            ("cutlass", "t4"): execute_cutlass_t4,
            ("cutlass", "a100"): execute_cutlass_a100,
            ("cutlass", "h100"): execute_cutlass_h100,
            ("cutlass", "b200"): execute_cutlass_b200,
            ("cutedsl", "t4"): execute_triton_t4,  # CUTE DSL uses Triton/Python
            ("cutedsl", "a100"): execute_triton_a100,
            ("cutedsl", "h100"): execute_triton_h100,
            ("cutedsl", "b200"): execute_triton_b200,
            ("mojo", "t4"): execute_mojo_t4,
            ("mojo", "a100"): execute_mojo_a100,
            ("mojo", "h100"): execute_mojo_h100,
            ("mojo", "b200"): execute_mojo_b200,
        }
        
        executor = executors.get((lang, gpu_tier))
        if not executor:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported language/GPU combination: {lang}/{gpu_tier}"
            )
        
        # Clamp timeout to safe range
        timeout = max(5, min(300, request.timeout))
        
        try:
            result = executor.remote(code=request.code, timeout_seconds=timeout)
            return ExecuteResponse(
                success=result["success"],
                stdout=result["stdout"],
                stderr=result["stderr"],
                execution_time=result["execution_time"],
            )
        except Exception as e:
            return ExecuteResponse(
                success=False,
                stdout="",
                stderr=f"Execution error: {str(e)}",
                execution_time=0,
            )

    @web_app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "ok", "app": "kernelide-executor"}

    @web_app.get("/")
    async def root():
        """Root endpoint with info."""
        return {
            "name": "KernelIDE Executor",
            "version": "1.0.0",
            "endpoints": {
                "/execute": "POST - Execute GPU code",
                "/health": "GET - Health check",
            },
            "supported_languages": ["cuda", "triton", "cutlass", "cutedsl", "mojo"],
            "supported_gpus": ["T4", "L4", "A10", "A100-40GB", "A100-80GB", "L40S", "H100", "H200", "B200"],
        }

    return web_app


@app.function(image=web_image)
@modal.asgi_app()
def api():
    """ASGI app entry point."""
    return create_web_app()


if __name__ == "__main__":
    print("Deploy this file with: modal deploy modal_executor.py")
    print("After deployment, your endpoint will be:")
    print("  https://<your-workspace>--kernelide-executor-api.modal.run")
