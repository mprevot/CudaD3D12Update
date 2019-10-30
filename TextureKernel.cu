#include <stdio.h>
#include "float_intrinsic2.h"
#include "ShaderStructs.h"
#include "helper_cuda.h"
#include "device_launch_parameters.h"

int iDivUp(int a, int b) { return a % b != 0 ? a / b + 1 : a / b; }

__global__ void TextureKernel(float *pixels, unsigned int width, unsigned int height, float time)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (y < height && x < width)
    {
		auto pos = (y * width + x) * 3;
		auto sint = __sinf(time) * 0.1f + 0.10f;
		auto sintAlt = (x / 32) % 2 == 0 ? 1.0f : sint;
		pixels[pos + 0] = sintAlt; //RED
		pixels[pos + 1] = 0; // (x + y) % 2 == 0 ? 1.0f : __sinf(time) * 0.25f + 0.75f; //GREEN
		pixels[pos + 2] = 0; // (x + y) % 2 == 0 ? 1.0f : 0.0f;						  //BLUE
		//pixels[pos + 0] = __sinf(time + 0.) * 0.5f + 0.5f;
		//pixels[pos + 1] = __sinf(time * 0.09) * 0.5f + 0.5f;
		//pixels[pos + 2] = __sinf(time + 2) * 0.5f + 0.5f;
    }
}

void RunKernel(size_t meshWidth, size_t meshHeight, float *texture_dev, cudaStream_t streamToRun, float animTime)
{
	//dim3 block(16, 16, 1);
	//dim3 grid(meshWidth / 16, meshHeight / 16, 1);
	auto unit = 32;
	dim3 threads(unit, unit);
	dim3 grid(iDivUp(meshWidth, unit), iDivUp(meshHeight, unit));
	TextureKernel <<<grid, threads, 0, streamToRun >>>(texture_dev, meshWidth, meshHeight, animTime);
	getLastCudaError("TextureKernel execution failed.\n");
}

