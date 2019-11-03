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

void RunKernel(size_t textureW, size_t textureH, cudaSurfaceObject_t surface, cudaStream_t streamToRun, float animTime);
void RunKernel(size_t textureW, size_t textureH, surfaceReference *surfaceRef, cudaStream_t streamToRun, float animTime);