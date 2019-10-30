#pragma once

#include "DX12CudaSample.h"
#include "ShaderStructs.h"
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::steady_clock::time_point TimePoint;

using namespace DirectX;

// Note that while ComPtr is used to manage the lifetime of resources on the CPU,
// it has no understanding of the lifetime of resources on the GPU. Apps must account
// for the GPU lifetime of resources to avoid destroying objects that may still be
// referenced by the GPU.
// An example of this can be found in the class method: OnDestroy().
using Microsoft::WRL::ComPtr;

class DX12CudaInterop : public DX12CudaSample
{
public:
	DX12CudaInterop(UINT width, UINT height, std::string name);

	virtual void OnInit();
	virtual void OnRender();
	virtual void OnDestroy();

private:
	// In this sample we overload the meaning of FrameCount to mean both the maximum
	// number of frames that will be queued to the GPU at a time, as well as the number
	// of back buffers in the DXGI swap chain. For the majority of applications, this
	// is convenient and works well. However, there will be certain cases where an
	// application may want to queue up more frames than there are back buffers
	// available.
	// It should be noted that excessive buffering of frames dependent on user input
	// may result in noticeable latency in your app.
	static const UINT FrameCount = 2;
	size_t TextureHeight, TextureWidth;

	ComPtr<IDXGIFactory4> dxgiFactory;
	ComPtr<ID3D12Debug> debugController;

	// Pipeline objects.
	D3D12_VIEWPORT m_viewport;
	CD3DX12_RECT m_scissorRect;
	ComPtr<IDXGISwapChain3> m_swapChain;
	ComPtr<ID3D12Device> m_device;
	ComPtr<ID3D12Resource> m_renderTargets[FrameCount];
	ComPtr<ID3D12CommandAllocator> m_commandAllocators[FrameCount];
	ComPtr<ID3D12CommandQueue>   m_commandQueue;
	ComPtr<ID3D12RootSignature>  m_rootSignature;
	ComPtr<ID3D12DescriptorHeap> m_rtvHeap;
	ComPtr<ID3D12DescriptorHeap> m_srvHeap;
	ComPtr<ID3D12PipelineState>  m_pipelineState;
	ComPtr<ID3D12GraphicsCommandList> m_commandList;
	UINT m_rtvDescriptorSize;
	
	// App resources.
	ComPtr<ID3D12Resource> m_vertexBuffer;
	ComPtr<ID3D12Resource> m_textureBuffer;
	D3D12_VERTEX_BUFFER_VIEW m_vertexBufferView;

	// Synchronization objects.
	UINT m_frameIndex;
	HANDLE m_fenceEvent;
	ComPtr<ID3D12Fence> m_fence;
	UINT64 m_fenceValues[FrameCount];

	TimePoint lastTimePoint;

	// CUDA objects
	cudaExternalMemoryHandleType m_externalMemoryHandleType;
	cudaExternalMemory_t	     m_externalMemory;
	cudaExternalSemaphore_t      m_externalSemaphore;
	cudaStream_t				 m_streamToRun;
	LUID						 m_dx12deviceluid;
	UINT						 m_cudaDeviceID;
	UINT						 m_nodeMask;
	float						 m_AnimTime;
	float timeStep{0.1};
	void *m_cudaDevVertptr{};

	void LoadPipeline();
	void InitCuda();
	void LoadAssets();
	void PopulateCommandList();
	void MoveToNextFrame();
	void WaitForGpu();
	cudaExternalSemaphoreWaitParams& GetWaitParamSemaphore(UINT64 fence);
};
