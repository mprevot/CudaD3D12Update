#pragma once

#include "stdafx.h"
#include <cuda_runtime.h>
#include "helper_cuda.h"

using namespace DirectX;

struct Vertex
{
	XMFLOAT3 position;
	XMFLOAT4 color;
};

struct TexVertex
{
	XMFLOAT3 position;
	XMFLOAT2 uv;
};

void RunKernel(size_t mesh_width, size_t mesh_height, cudaSurfaceObject_t cudaDevVertptr, cudaStream_t streamToRun, float animTime, UINT8 nChannels);