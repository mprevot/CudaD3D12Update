#include "stdafx.h"
#include "simpleD3D12.h"
#include <aclapi.h>
#include <sstream>

using namespace std;

//DXGI_FORMAT_R16G16B16A16_FLOAT
//DXGI_FORMAT_R8G8B8A8_UNORM
//DXGI_FORMAT_R10G10B10A2_UNORM
#define DGXIFormat DXGI_FORMAT_R10G10B10A2_UNORM

class WindowsSecurityAttributes {
protected:
	SECURITY_ATTRIBUTES m_winSecurityAttributes;
	PSECURITY_DESCRIPTOR m_winPSecurityDescriptor;

public:
	WindowsSecurityAttributes();
	~WindowsSecurityAttributes();
	SECURITY_ATTRIBUTES * operator&();
};

WindowsSecurityAttributes::WindowsSecurityAttributes()
{
	m_winPSecurityDescriptor = (PSECURITY_DESCRIPTOR)calloc(1, SECURITY_DESCRIPTOR_MIN_LENGTH + 2 * sizeof(void**));
	assert(m_winPSecurityDescriptor != (PSECURITY_DESCRIPTOR)NULL);

	PSID *ppSID = (PSID *)((PBYTE)m_winPSecurityDescriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
	PACL *ppACL = (PACL *)((PBYTE)ppSID + sizeof(PSID *));

	InitializeSecurityDescriptor(m_winPSecurityDescriptor, SECURITY_DESCRIPTOR_REVISION);

	SID_IDENTIFIER_AUTHORITY sidIdentifierAuthority = SECURITY_WORLD_SID_AUTHORITY;
	AllocateAndInitializeSid(&sidIdentifierAuthority, 1, SECURITY_WORLD_RID, 0, 0, 0, 0, 0, 0, 0, ppSID);

	EXPLICIT_ACCESS explicitAccess;
	ZeroMemory(&explicitAccess, sizeof(EXPLICIT_ACCESS));
	explicitAccess.grfAccessPermissions = STANDARD_RIGHTS_ALL | SPECIFIC_RIGHTS_ALL;
	explicitAccess.grfAccessMode = SET_ACCESS;
	explicitAccess.grfInheritance = INHERIT_ONLY;
	explicitAccess.Trustee.TrusteeForm = TRUSTEE_IS_SID;
	explicitAccess.Trustee.TrusteeType = TRUSTEE_IS_WELL_KNOWN_GROUP;
	explicitAccess.Trustee.ptstrName = (LPTSTR)*ppSID;

	SetEntriesInAcl(1, &explicitAccess, NULL, ppACL);

	SetSecurityDescriptorDacl(m_winPSecurityDescriptor, TRUE, *ppACL, FALSE);

	m_winSecurityAttributes.nLength = sizeof(m_winSecurityAttributes);
	m_winSecurityAttributes.lpSecurityDescriptor = m_winPSecurityDescriptor;
	m_winSecurityAttributes.bInheritHandle = TRUE;
}

WindowsSecurityAttributes::~WindowsSecurityAttributes()
{
	PSID* ppSID = (PSID*)((PBYTE)m_winPSecurityDescriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
	PACL* ppACL = (PACL*)((PBYTE)ppSID + sizeof(PSID*));

	if (*ppSID)
		FreeSid(*ppSID);
	if (*ppACL)
		LocalFree(*ppACL);
	free(m_winPSecurityDescriptor);
}

SECURITY_ATTRIBUTES *
WindowsSecurityAttributes::operator&() { return &m_winSecurityAttributes; }

DX12CudaInterop::DX12CudaInterop(UINT width, UINT height, std::string name) :
	DX12CudaSample(width, height, name),
	m_frameIndex(0),
	m_scissorRect(0, 0, static_cast<LONG>(width), static_cast<LONG>(height)),
	m_fenceValues{},
	m_rtvDescriptorSize(0)
{
	m_viewport = { 0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height) };
	m_AnimTime = 1.0f;
}

void DX12CudaInterop::OnInit()
{
	LoadPipeline();
	InitCuda();
	LoadAssets();
}

// Load the rendering pipeline dependencies.
void DX12CudaInterop::LoadPipeline()
{
	UINT dxgiFactoryFlags{};
#if defined(_DEBUG)
	// Enable the debug layer (requires the Graphics Tools "optional feature").
	// NOTE: Enabling the debug layer after device creation will invalidate the active device.
	if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController))))
	{
		debugController->EnableDebugLayer();
		dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
	}
#endif

	ComPtr<IDXGIFactory4> factory;
	ThrowIfFailed(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)));

	ComPtr<IDXGIAdapter1> hardwareAdapter;
	GetHardwareAdapter(factory.Get(), &hardwareAdapter);

	ThrowIfFailed(D3D12CreateDevice(hardwareAdapter.Get(), D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&m_device)));
	DXGI_ADAPTER_DESC1 desc;
	hardwareAdapter->GetDesc1(&desc);
	m_dx12deviceluid = desc.AdapterLuid;

	// Describe and create the command queue.
	D3D12_COMMAND_QUEUE_DESC queueDesc{};
	queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
	queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

	ThrowIfFailed(m_device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_commandQueue)));

	// Describe and create the swap chain.
	DXGI_SWAP_CHAIN_DESC1 swapChainDesc{};
	swapChainDesc.BufferCount = FrameCount;
	swapChainDesc.Width = m_width;
	swapChainDesc.Height = m_height;
	swapChainDesc.Format = DGXIFormat;
	swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	swapChainDesc.SampleDesc.Count = 1;

	ComPtr<IDXGISwapChain1> swapChain{};
	ThrowIfFailed(factory->CreateSwapChainForHwnd(m_commandQueue.Get(), Win32Application::GetHwnd(),
		&swapChainDesc, nullptr, nullptr, &swapChain));

	// This sample does not support fullscreen transitions.
	ThrowIfFailed(factory->MakeWindowAssociation(Win32Application::GetHwnd(), DXGI_MWA_NO_ALT_ENTER));

	ThrowIfFailed(swapChain.As(&m_swapChain));
	m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();

	// Create descriptor heaps.
	{
		// Describe and create a render target view (RTV) descriptor heap.
		D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc{};
		rtvHeapDesc.NumDescriptors = FrameCount;
		rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
		rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
		ThrowIfFailed(m_device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&m_rtvHeap)));
		NAME_D3D12_OBJECT(m_rtvHeap);

		// Describe and create a shader resource view (SRV) heap for the texture.
		D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc{};
		srvHeapDesc.NumDescriptors = 1;
		srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
		srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
		ThrowIfFailed(m_device->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&m_srvHeap)));
		NAME_D3D12_OBJECT(m_srvHeap);

		m_rtvDescriptorSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
	}

	// Create frame resources.
	{
		CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(m_rtvHeap->GetCPUDescriptorHandleForHeapStart());

		// Create a RTV and a command allocator for each frame.
		for (UINT n = 0; n < FrameCount; n++)
		{
			ThrowIfFailed(m_swapChain->GetBuffer(n, IID_PPV_ARGS(&m_renderTargets[n])));
			m_device->CreateRenderTargetView(m_renderTargets[n].Get(), nullptr, rtvHandle);
			rtvHandle.Offset(1, m_rtvDescriptorSize);
			ThrowIfFailed(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_commandAllocators[n])));
		}
	}
}

void DX12CudaInterop::InitCuda()
{
	int num_cuda_devices = 0;
	CheckCudaErrors(cudaGetDeviceCount(&num_cuda_devices));

	if (!num_cuda_devices)
		throw std::exception("No CUDA Devices found");
	for (UINT devId = 0; devId < num_cuda_devices; devId++)
	{
		cudaDeviceProp devProp{};
		CheckCudaErrors(cudaGetDeviceProperties(&devProp, devId));
		const auto cmp1 = memcmp(&m_dx12deviceluid.LowPart, devProp.luid, sizeof(m_dx12deviceluid.LowPart)) == 0;
		const auto cmp2 = memcmp(&m_dx12deviceluid.HighPart, devProp.luid + sizeof(m_dx12deviceluid.LowPart), sizeof(m_dx12deviceluid.HighPart)) == 0;
		if (cmp1 && cmp2)
        {
			CheckCudaErrors(cudaSetDevice(devId));
			m_cudaDeviceID = devId;
			m_nodeMask = devProp.luidDeviceNodeMask;
			CheckCudaErrors(cudaStreamCreate(&m_streamToRun));
			printf("CUDA Device Used [%d] %s\n", devId, devProp.name);
			break;
		}
	}
}

inline void Open(string path)
{
	replace(path.begin(), path.end(), '/', '\\');
	ShellExecute(0, 0, path.c_str(), 0, 0, SW_SHOW);
}

// Load the sample assets.
void DX12CudaInterop::LoadAssets()
{
	// Create a root signature.
	{
		D3D12_FEATURE_DATA_ROOT_SIGNATURE featureData{};
		featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;
		if (FAILED(m_device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &featureData, sizeof(featureData))))
			featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_0;

		CD3DX12_DESCRIPTOR_RANGE1 ranges[1];
		ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC);

		CD3DX12_ROOT_PARAMETER1 rootParameters[1];
		rootParameters[0].InitAsDescriptorTable(1, &ranges[0], D3D12_SHADER_VISIBILITY_PIXEL);
		
		D3D12_STATIC_SAMPLER_DESC sampler{};
		sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
		sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
		sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
		sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
		sampler.MipLODBias = 0;
		sampler.MaxAnisotropy = 0;
		sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
		sampler.BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
		sampler.MinLOD = 0.0f;
		sampler.MaxLOD = D3D12_FLOAT32_MAX;
		sampler.ShaderRegister = 0;
		sampler.RegisterSpace = 0;
		sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

		CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSignatureDesc;
		rootSignatureDesc.Init_1_1(_countof(rootParameters), rootParameters, 1, &sampler,
			D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

		ComPtr<ID3DBlob> signature;
		ComPtr<ID3DBlob> error;
		ThrowIfFailed(D3DX12SerializeVersionedRootSignature(&rootSignatureDesc, featureData.HighestVersion, &signature, &error));
		ThrowIfFailed(m_device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_rootSignature)));
		NAME_D3D12_OBJECT(m_device);
		NAME_D3D12_OBJECT(m_rootSignature);
	}

	// Create the pipeline state, which includes compiling and loading shaders.
	{
		ComPtr<ID3DBlob> vertexShader;
		ComPtr<ID3DBlob> pixelShader;

#if defined(_DEBUG)
		// Enable better shader debugging with the graphics debugging tools.
		UINT compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
		UINT compileFlags = 0;
#endif
		
		std::wstring filePath = GetAssetFullPath("texShader.hlsl");
		LPCWSTR result = filePath.c_str();
		ThrowIfFailed(D3DCompileFromFile(result, 0, 0, "VSMain", "vs_5_0", compileFlags, 0, &vertexShader, 0));
		ThrowIfFailed(D3DCompileFromFile(result, 0, 0, "PSMain", "ps_5_0", compileFlags, 0, &pixelShader, 0));
		
		// Define the vertex input layout.
		D3D12_INPUT_ELEMENT_DESC inputElementDescs[] =
		{
			{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
			{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
		};

		// Describe and create the graphics pipeline state object (PSO).
		D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
		psoDesc.InputLayout = { inputElementDescs, _countof(inputElementDescs) };
		psoDesc.pRootSignature = m_rootSignature.Get();
		psoDesc.VS = CD3DX12_SHADER_BYTECODE(vertexShader.Get());
		psoDesc.PS = CD3DX12_SHADER_BYTECODE(pixelShader.Get());
		psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
		psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
		psoDesc.DepthStencilState.DepthEnable = FALSE;
		psoDesc.DepthStencilState.StencilEnable = FALSE;
		psoDesc.SampleMask = UINT_MAX;
		psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
		psoDesc.NumRenderTargets = 1;
		psoDesc.RTVFormats[0] = DGXIFormat;
		psoDesc.SampleDesc.Count = 1;
		ThrowIfFailed(m_device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_pipelineState)));
		NAME_D3D12_OBJECT(m_pipelineState);
	}

	ThrowIfFailed(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_commandAllocators[m_frameIndex].Get(), m_pipelineState.Get(), IID_PPV_ARGS(&m_commandList)));

	// Create the vertex buffer.
	ComPtr<ID3D12Resource> vertexBufferUpload{};
	{
		constexpr auto y = 1.0f;// *m_aspectRatio;
		constexpr auto x = 1.0f;
		TexVertex quadVertices[] =
		{
			{ {-x,-y, 0.0f }, { 0.0f, 0.0f } },
			{ {-x, y, 0.0f }, { 0.0f, 1.0f } },
			{ {x, -y, 0.0f }, { 1.0f, 0.0f } },
			{ {x,  y, 0.0f }, { 1.0f, 1.0f } },
		};

		constexpr auto vertexBufferSize = sizeof(quadVertices);
		ThrowIfFailed(m_device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE,
			&CD3DX12_RESOURCE_DESC::Buffer(vertexBufferSize), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&m_vertexBuffer)));
		ThrowIfFailed(m_device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE,
			&CD3DX12_RESOURCE_DESC::Buffer(vertexBufferSize), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&vertexBufferUpload)));
		NAME_D3D12_OBJECT(m_vertexBuffer);

		// Copy data to the intermediate upload heap and then schedule a copy 
		// from the upload heap to the vertex buffer.
		D3D12_SUBRESOURCE_DATA vertexData{};
		vertexData.pData = &quadVertices[0];
		vertexData.RowPitch = vertexBufferSize;
		vertexData.SlicePitch = vertexData.RowPitch;

		UpdateSubresources<1>(m_commandList.Get(), m_vertexBuffer.Get(), vertexBufferUpload.Get(), 0, 0, 1, &vertexData);
		m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_vertexBuffer.Get(),
			D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER));

		// Initialize the vertex buffer view.
		m_vertexBufferView.BufferLocation = m_vertexBuffer->GetGPUVirtualAddress();
		m_vertexBufferView.StrideInBytes = sizeof(TexVertex);
		m_vertexBufferView.SizeInBytes = sizeof(quadVertices);
	}

	// Texture
	{
		TextureChannels = 4;
		TextureWidth = m_width;
		TextureHeight = m_height;
		const auto textureSurface = TextureWidth * TextureHeight;
		const auto texturePixels = textureSurface * TextureChannels;
		const auto textureSizeBytes = sizeof(float)* texturePixels;

		const auto texFormat = TextureChannels == 4 ? DXGI_FORMAT_R32G32B32A32_FLOAT : DXGI_FORMAT_R32G32B32_FLOAT;
		const auto texDesc = CD3DX12_RESOURCE_DESC::Tex2D(texFormat, TextureWidth, TextureHeight, 1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS);
		ThrowIfFailed(m_device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_SHARED,
			&texDesc, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, nullptr, IID_PPV_ARGS(&TextureArray)));
		NAME_D3D12_OBJECT(TextureArray);

		D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{};
		srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srvDesc.Format = texDesc.Format;
		srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
		srvDesc.Texture2D.MipLevels = texDesc.MipLevels;
		m_device->CreateShaderResourceView(TextureArray.Get(), &srvDesc, m_srvHeap->GetCPUDescriptorHandleForHeapStart());
		NAME_D3D12_OBJECT(m_srvHeap);

		HANDLE sharedHandle{};
		WindowsSecurityAttributes secAttr{};
		ThrowIfFailed(m_device->CreateSharedHandle(TextureArray.Get(), &secAttr, GENERIC_ALL, 0, &sharedHandle));
		const auto texAllocInfo = m_device->GetResourceAllocationInfo(m_nodeMask, 1, &texDesc);

		cudaExternalMemoryHandleDesc cuExtmemHandleDesc{};
		cuExtmemHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Heap;
		cuExtmemHandleDesc.handle.win32.handle = sharedHandle;
		cuExtmemHandleDesc.size = texAllocInfo.SizeInBytes;
		cuExtmemHandleDesc.flags = cudaExternalMemoryDedicated;
		CheckCudaErrors(cudaImportExternalMemory(&m_externalMemory, &cuExtmemHandleDesc));

		cudaExternalMemoryMipmappedArrayDesc cuExtmemMipDesc{};
		cuExtmemMipDesc.extent = make_cudaExtent(texDesc.Width, texDesc.Height, 0);
		cuExtmemMipDesc.formatDesc = cudaCreateChannelDesc<float4>();
		cuExtmemMipDesc.numLevels = 1;
		cuExtmemMipDesc.flags = cudaArraySurfaceLoadStore;
		
		cudaMipmappedArray_t cuMipArray{};
		CheckCudaErrors(cudaExternalMemoryGetMappedMipmappedArray(&cuMipArray, m_externalMemory, &cuExtmemMipDesc));

		cudaArray_t cuArray{};
		CheckCudaErrors(cudaGetMipmappedArrayLevel(&cuArray, cuMipArray, 0));
		
		cudaResourceDesc cuResDesc{};
		cuResDesc.resType = cudaResourceTypeArray;
		cuResDesc.res.array.array = cuArray;
		checkCudaErrors(cudaCreateSurfaceObject(&cuSurface, &cuResDesc));
		 
		m_AnimTime = 1.0f;
		UpdateCudaSurface();
		
		CheckCudaErrors(cudaStreamSynchronize(m_streamToRun));
	}

	// Close and execute commands
	{
		ThrowIfFailed(m_commandList->Close());
		ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
		m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);
	}

	// Create synchronization objects and wait until assets have been uploaded to the GPU.
	{
		ThrowIfFailed(m_device->CreateFence(m_fenceValues[m_frameIndex], D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&m_fence)));
		NAME_D3D12_OBJECT(m_fence);

		cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc{};
		memset(&externalSemaphoreHandleDesc, 0, sizeof(externalSemaphoreHandleDesc));
		WindowsSecurityAttributes windowsSecurityAttributes;
		LPCWSTR name{};
		HANDLE sharedHandle{};
		externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
		m_device->CreateSharedHandle(m_fence.Get(), &windowsSecurityAttributes, GENERIC_ALL, name, &sharedHandle);
		externalSemaphoreHandleDesc.handle.win32.handle = sharedHandle;
		externalSemaphoreHandleDesc.flags = 0;
		CheckCudaErrors(cudaImportExternalSemaphore(&m_externalSemaphore, &externalSemaphoreHandleDesc));
		m_fenceValues[m_frameIndex]++;

		m_fenceEvent = CreateEvent(nullptr, false, false, nullptr);
		if (m_fenceEvent == nullptr)
			ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));
		WaitForGpu();
	}
}

void DX12CudaInterop::UpdateCudaSurface()
{
	//RunKernel(TextureWidth, TextureHeight, &cuSurfaceRef, m_streamToRun, m_AnimTime);
	RunKernel(TextureWidth, TextureHeight, cuSurface, m_streamToRun, m_AnimTime);
}

// Render the scene.
void DX12CudaInterop::OnRender()
{
	auto currentPoint = Clock::now();
	auto period = std::chrono::duration_cast<std::chrono::duration<double>>(currentPoint - lastTimePoint);
	lastTimePoint = Clock::now();
	std::stringstream s;
	s << " Freq: " << 1.0f / period.count() << " Hz";
	SetCustomWindowText(s.str().c_str());
	
	// Record all the commands we need to render the scene into the command list.
	PopulateCommandList();

	// Execute the command list.
	ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
	m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

	// Present the frame.
	ThrowIfFailed(m_swapChain->Present(1, 0));

	// Schedule a Signal command in the queue.
	const UINT64 currentFenceValue = m_fenceValues[m_frameIndex];
	ThrowIfFailed(m_commandQueue->Signal(m_fence.Get(), currentFenceValue));

	MoveToNextFrame();
}

void DX12CudaInterop::OnDestroy()
{
	// Ensure that the GPU is no longer referencing resources that are about to be
	// cleaned up by the destructor.
	WaitForGpu();
	CheckCudaErrors(cudaDestroyExternalSemaphore(m_externalSemaphore));
	CheckCudaErrors(cudaDestroyExternalMemory(m_externalMemory));
	CloseHandle(m_fenceEvent);
}

void DX12CudaInterop::PopulateCommandList()
{
	// Command list allocators can only be reset when the associated 
	// command lists have finished execution on the GPU; apps should use 
	// fences to determine GPU execution progress.
	ThrowIfFailed(m_commandAllocators[m_frameIndex]->Reset());

	// However, when ExecuteCommandList() is called on a particular command 
	// list, that command list can then be reset at any time and must be before 
	// re-recording.
	ThrowIfFailed(m_commandList->Reset(m_commandAllocators[m_frameIndex].Get(), m_pipelineState.Get()));

	m_commandList->SetGraphicsRootSignature(m_rootSignature.Get());
	
	ID3D12DescriptorHeap* ppHeaps[] = { m_srvHeap.Get() };
	m_commandList->SetDescriptorHeaps(_countof(ppHeaps), ppHeaps);
	m_commandList->SetGraphicsRootDescriptorTable(0, m_srvHeap->GetGPUDescriptorHandleForHeapStart());

	// Set necessary state.
	m_commandList->RSSetViewports(1, &m_viewport);
	m_commandList->RSSetScissorRects(1, &m_scissorRect);

	// Indicate that the back buffer will be used as a render target.
	m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_renderTargets[m_frameIndex].Get(),
		D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET));

	CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(m_rtvHeap->GetCPUDescriptorHandleForHeapStart(), m_frameIndex, m_rtvDescriptorSize);
	m_commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);

	// Record commands.
	const float clearColor[] = { 0.0f, 0.2f, 0.4f, 1.0f };
	m_commandList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
	m_commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
	m_commandList->IASetVertexBuffers(0, 1, &m_vertexBufferView);
	m_commandList->DrawInstanced(TextureHeight*TextureWidth, 1, 0, 0);

	// Indicate that the back buffer will now be used to present.
	m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_renderTargets[m_frameIndex].Get(),
		D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));

	ThrowIfFailed(m_commandList->Close());
}

void DX12CudaInterop::WaitForGpu()
{
	// Schedule a Signal command in the queue.
	ThrowIfFailed(m_commandQueue->Signal(m_fence.Get(), m_fenceValues[m_frameIndex]));

	// Wait until the fence has been processed.
	ThrowIfFailed(m_fence->SetEventOnCompletion(m_fenceValues[m_frameIndex], m_fenceEvent));
	WaitForSingleObjectEx(m_fenceEvent, INFINITE, FALSE);

	// Increment the fence value for the current frame.
	m_fenceValues[m_frameIndex]++;
}

void DX12CudaInterop::MoveToNextFrame()
{
	const UINT64 currentFenceValue = m_fenceValues[m_frameIndex];
	
	cudaExternalSemaphoreWaitParams externalSemaphoreWaitParams{};
	externalSemaphoreWaitParams.params.fence.value = currentFenceValue;
	CheckCudaErrors(cudaWaitExternalSemaphoresAsync(&m_externalSemaphore, &externalSemaphoreWaitParams, 1, m_streamToRun));

	m_AnimTime += .05;
	UpdateCudaSurface();

	m_fenceValues[m_frameIndex] = currentFenceValue + 1;
	
	cudaExternalSemaphoreSignalParams externalSemaphoreSignalParams{};
	externalSemaphoreSignalParams.params.fence.value = m_fenceValues[m_frameIndex];
	CheckCudaErrors(cudaSignalExternalSemaphoresAsync(&m_externalSemaphore, &externalSemaphoreSignalParams, 1, m_streamToRun));

	// Update the frame index.
	m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();
	// If the next frame is not ready to be rendered yet, wait until it is ready.
	if (m_fence->GetCompletedValue() < m_fenceValues[m_frameIndex])
	{
		ThrowIfFailed(m_fence->SetEventOnCompletion(m_fenceValues[m_frameIndex], m_fenceEvent));
		WaitForSingleObjectEx(m_fenceEvent, INFINITE, FALSE);
	}
	// Set the fence value for the next frame.
	m_fenceValues[m_frameIndex] = currentFenceValue + 2;
}
