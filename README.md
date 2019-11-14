# CudaD3D12Update

A sample that demonstrate a DirectX 12 texture updated by cuda, through a cudaArray / surface2D.

`float_intrinsic.h` demonstrates also how to remove Intellisense's and Resharper's "symbol not found" without interfering with nvcc, as well as adding intrinsic functions with their documentation.

## requirements

### software
- Windows 10 (1903 recommended)
- Visual Studio 2019 (workloads: desktop dev with c++, game dev with c++)
- cuda 10 (https://developer.nvidia.com/cuda-downloads)

### hardware
- nvidia GPU with 7.5 compute calpabilities (you can adjust the achitecture target in the project eventually).

## feedback 
Contact: mprevot@freebsd.org
