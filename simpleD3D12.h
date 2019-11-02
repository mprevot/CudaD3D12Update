#pragma once

#include "DX12CudaSample.h"
#include "ShaderStructs.h"
#include <chrono>
#include <sstream>
#include <string>
#include <algorithm>

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::steady_clock::time_point TimePoint;

using namespace std;
using namespace DirectX;
using Microsoft::WRL::ComPtr;

#define CheckCudaErrors(val) Check((val), #val, __FUNCTION__, __FILE__, __LINE__)

typedef void(__stdcall* MessageChangedCallback)(const wchar_t* string);

class DX12CudaInterop : public DX12CudaSample
{
public:
	DX12CudaInterop(UINT width, UINT height, std::string name);
	virtual void OnInit();
	virtual void OnRender();
	virtual void OnDestroy();
	
	template <class T>
	auto Check(T result, const char* func, const char* caller, const char* file, int line) -> void;
	template <class ... T>
	auto LogMessage(T&& ... args) -> void;

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

	vector<wstring> Messages{};
	MessageChangedCallback LogMessageChangedCallback{};

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
	ComPtr<ID3D12Resource> TextureArray;
	D3D12_VERTEX_BUFFER_VIEW m_vertexBufferView;

	// Synchronization objects.
	UINT m_frameIndex;
	HANDLE m_fenceEvent;
	ComPtr<ID3D12Fence> m_fence;
	UINT64 m_fenceValues[FrameCount];

	TimePoint lastTimePoint;

	// CUDA objects
	cudaExternalMemoryHandleType m_externalMemoryHandleType;
	cudaExternalMemory_t m_externalMemory;
	cudaExternalSemaphore_t m_externalSemaphore;
	cudaStream_t m_streamToRun;
	LUID m_dx12deviceluid;
	UINT m_cudaDeviceID;
	UINT m_nodeMask;
	float m_AnimTime;
	float timeStep{0.1};
	
	float* cuTexArray{};
	size_t cuTexArraySize;
	cudaMipmappedArray_t cuMipArray{};
	cudaArray_t cuArray{};
	cudaSurfaceObject_t cuSurface{};

	UINT8 TextureChannels;
	size_t TextureSize_dev{};
	
	void LoadPipeline();
	void InitCuda();
	void LoadAssets();
	void UpdateCudaSurface();
	void PopulateCommandList();
	void MoveToNextFrame();
	void WaitForGpu();
};

template <class T>
auto DX12CudaInterop::Check(T result, const char* func, const char* caller, const char* file, int line) -> void
{
	if (result)
	{
		wstringstream o;
		auto f = string(file);
		replace(f.begin(), f.end(), '\\', '/');
		auto justfile = string(f.substr(f.find_last_of('/') + 1));
		o << caller << "(): " << func << " at " << justfile.c_str() << ":" << line << " [" << cudaGetErrorName((cudaError_t)result) << "]";
		LogMessage(L"%s\n", o.str().c_str());
	}
}

template<class ...T>
auto DX12CudaInterop::LogMessage(T&&... args) -> void
{
	//if (LogMessageChangedCallback != nullptr)
	{
		wchar_t updatedMessage[4096];
		swprintf_s(updatedMessage, forward<T>(args)...);
		Messages.push_back(*new wstring(updatedMessage));
		//LogMessageChangedCallback(updatedMessage);
	}
}
