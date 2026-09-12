"""Build: pip install --no-build-isolation ./20-torch-extension

--no-build-isolation so the extension compiles against the torch already
installed, not a fresh copy pip would otherwise download into a temporary
environment. The CUDA toolkit's major version must match torch's (torch
refuses to build otherwise); on Windows, run from a shell where cl.exe is on
PATH (a Visual Studio developer prompt, or after vcvars64.bat).
"""

import os
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# sm_75 (Turing: GTX 16xx / T4) unless told otherwise. The kernels use __dp4a,
# which needs 6.1+.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")

windows = sys.platform == "win32"
# The binding includes torch's headers, which need C++20 as of torch 2.14. The
# kernel file includes none of them, so nvcc stays on C++17 -- the split between
# ops.cpp and q4_kernels.cu is what makes that possible.
cxx_flags = ["/O2", "/std:c++20", "/Zc:preprocessor"] if windows else ["-O3", "-std=c++20"]
nvcc_flags = ["-O3", "-std=c++17"]
if windows:
    nvcc_flags.append("-Xcompiler=/Zc:preprocessor")

setup(
    name="cuda_portfolio_ops",
    version="0.1.0",
    description="Q4_0 quantized linear layers with hand-written CUDA kernels",
    packages=["cuda_portfolio_ops"],
    ext_modules=[
        CUDAExtension(
            name="cuda_portfolio_ops._C",
            sources=["csrc/ops.cpp", "csrc/q4_kernels.cu"],
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
    install_requires=["torch"],
)
