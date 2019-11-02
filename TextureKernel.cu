#include <stdio.h>
#include "float_intrinsic2.h"
#include "ShaderStructs.h"
#include "helper_cuda.h"
#include "device_launch_parameters.h"

int iDivUp(int a, int b) { return a % b != 0 ? a / b + 1 : a / b; }

__global__ void UpdateSurface(cudaSurfaceObject_t surface, unsigned int width, unsigned int height, float time, UINT8 nChannels)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (y >= height | x >= width) return;
	auto xVar = (float)x / (float)width;
	auto yVar = (float)y / (float)height;
	
	auto pos = (y * width + x) * nChannels;
	auto cost = cosf(time) * 0.5f + 0.5f;
	auto costx = cosf(time) * 0.5f + xVar;
	auto costy = cosf(time) * 0.5f + yVar;
	auto costxx = (cosf(time) * 0.5f + 0.5f) * width;
	auto costyy = (cosf(time) * 0.5f + 0.5f)* height;

	float4 pixelRGBA{};
	
	if (x == width-1)
	{
		pixelRGBA.x = costx;
		pixelRGBA.y = 0;
		pixelRGBA.z = 0;
		surf2Dwrite(pixelRGBA, surface, x, y);
		return;
	}
	//if (x%3 == 0)
	//{
	//	pixelRGBA.x = cost;
	//	pixelRGBA.y = 0;
	//	pixelRGBA.z = 0;
	//	surf2Dwrite(pixelRGBA, surface, x, y);
	//	return;
	//}
	//if (x< costxx)
	//{
	//	pixelRGBA.x = 1;
	//	pixelRGBA.y = 0;
	//	pixelRGBA.z = 0;
	//	surf2Dwrite(pixelRGBA, surface, x, y);
	//	return;
	//}
	pixelRGBA.x = costx;
	pixelRGBA.y = 0;
	pixelRGBA.z = costx*0.3;
	surf2Dwrite(pixelRGBA, surface, x, y);

	//auto sintAlt = x/16 % 2 == 0 ? 1.0f : costx;
	//auto sintAlt2 = y % 2 == 0 ? 1.0f : costx;
	//pixels[pos + 0] = __sinf(time + 0.) * 0.5f + 0.5f;
	//pixels[pos + 1] = __sinf(time * 0.09) * 0.5f + 0.5f;
	//pixels[pos + 2] = __sinf(time + 2) * 0.5f + 0.5f;
}

void RunKernel(size_t textureW, size_t textureH, cudaSurfaceObject_t surface, cudaStream_t streamToRun, float animTime, UINT8 nChannels)
{
	//dim3 threads(16, 16, 1);
	//dim3 grid(meshWidth / 16, meshHeight / 16, 1);
	auto unit = 16;
	dim3 threads(unit, unit);
	dim3 grid(iDivUp(textureW, unit), iDivUp(textureH, unit));
	UpdateSurface <<<grid, threads, 0, streamToRun >>>(surface, textureW, textureH, animTime, nChannels);
	getLastCudaError("TextureKernel execution failed.\n");
}

