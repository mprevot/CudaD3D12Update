#include <stdio.h>
#include "float_intrinsic2.h"
#include "ShaderStructs.h"
#include "helper_cuda.h"
#include "device_launch_parameters.h"

int iDivUp(int a, int b) { return a % b != 0 ? a / b + 1 : a / b; }

__global__ void UpdateSurface(cudaSurfaceObject_t surface, unsigned int width, unsigned int height, float time)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (y >= height | x >= width) return;
	
	auto xVar = (float)x / (float)width;
	auto yVar = (float)y / (float)height;
	auto costx = __cosf(time) * 0.5f + xVar;
	auto cost = cosf(time) * 0.5f + 0.5f;
	auto costy = cosf(time) * 0.5f + yVar;
	auto costxx = (cosf(time) * 0.5f + 0.5f) * width;
	auto costyy = (cosf(time) * 0.5f + 0.5f) * height;
	
	float4 pixel{};
	if (y == 0)
		pixel = make_float4(1, 0, 0, 1);
	else if (y == height - 1)
		pixel = make_float4(1, 0, 1, 1);
	else if (x%10 == 0)
	{
		if(x>width/2)
			pixel = make_float4(0.1, 0.5, costx * 1, 1);
		else
			pixel = make_float4(costx * 1, 0.1, 0.2, 1);
	}
	else
		pixel = make_float4(costx * 0.2, costx * 0.4, costx * 0.6, 1);
	surf2Dwrite(pixel, surface, x, y);
}

void RunKernel(size_t textureW, size_t textureH, cudaSurfaceObject_t surface, cudaStream_t streamToRun, float animTime)
{
	auto unit = 16;
	dim3 threads(unit, unit);
	dim3 grid(iDivUp(textureW, unit), iDivUp(textureH, unit));
	UpdateSurface <<<grid, threads, 0, streamToRun >>>(surface, textureW, textureH, animTime);
	getLastCudaError("TextureKernel execution failed.\n");
}

