/*******************************************************************************
 * Copyright 2025 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files(the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions :
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 ******************************************************************************/

#include "NNERuntimeOpenVINOModule.h"

#include "NNERuntimeOpenVINOCpu.h"
#include "NNERuntimeOpenVINONpu.h"
#include "NNERuntimeOpenVINOGpu.h"
#include "NNE.h"
#include "NNERuntimeOpenVINOCommon.h"

#include "Modules/ModuleManager.h"
#include "Interfaces/IPluginManager.h"

IMPLEMENT_MODULE(FNNERuntimeOpenVINO, NNERuntimeOpenVINO)

DEFINE_LOG_CATEGORY(LogNNERuntimeOpenVINO);

void FNNERuntimeOpenVINO::StartupModule()
{
	UE_LOG(LogNNERuntimeOpenVINO, Display, TEXT("Loaded NNERuntimeOpenVINO"));

	if (!LoadDLL())
	{
		UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Failed to load the OpenVINO DLLs."));
		return;
	}

	ov_version_t OVVersion{};
	if (ov_get_openvino_version(&OVVersion))
	{
		UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Failed to get OpenVINO version."));
		return;
	}

	UE_LOG(LogNNERuntimeOpenVINO, Display, TEXT("Using OpenVINO %s %s"), ANSI_TO_TCHAR(OVVersion.buildNumber), ANSI_TO_TCHAR(OVVersion.description));

	ov_version_free(&OVVersion);

	if (ov_core_create(&OVCore))
	{
		UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Failed to create an OpenVINO instance."));
		return;
	}

	LogDevices();

#ifdef OPENVINO_CPU_PLUGIN
	// NNE runtime ORT Cpu startup
	NNERuntimeOpenVINOCpu = NewObject<UNNERuntimeOpenVINOCpu>();
	if (NNERuntimeOpenVINOCpu.IsValid())
	{
		TWeakInterfacePtr<INNERuntime> RuntimeCPUInterface(NNERuntimeOpenVINOCpu.Get());

		NNERuntimeOpenVINOCpu->AddToRoot();
		UE::NNE::RegisterRuntime(RuntimeCPUInterface);
	}
#endif

#ifdef OPENVINO_NPU_PLUGIN
	if (SupportsDevice(*OVCore, TEXT("NPU")))
	{
		// NNE runtime ORT Npu startup
		NNERuntimeOpenVINONpu = NewObject<UNNERuntimeOpenVINONpu>();
		if (NNERuntimeOpenVINONpu.IsValid())
		{
			TWeakInterfacePtr<INNERuntime> RuntimeNPUInterface(NNERuntimeOpenVINONpu.Get());

			NNERuntimeOpenVINONpu->AddToRoot();
			UE::NNE::RegisterRuntime(RuntimeNPUInterface);
		}
	}
	else
	{
		UE_LOG(LogNNERuntimeOpenVINO, Warning, TEXT("No NPU device found, INNERuntimeNPU will be unavailable."));
	}
#endif

#ifdef OPENVINO_GPU_PLUGIN
	if (SupportsDevice(*OVCore, TEXT("GPU")))
	{
		// NNE runtime ORT Gpu startup
		NNERuntimeOpenVINOGpu = NewObject<UNNERuntimeOpenVINOGpu>();
		if (NNERuntimeOpenVINONpu.IsValid())
		{
			TWeakInterfacePtr<INNERuntime> RuntimeGPUInterface(NNERuntimeOpenVINOGpu.Get());

			NNERuntimeOpenVINOGpu->AddToRoot();
			UE::NNE::RegisterRuntime(RuntimeGPUInterface);
		}
	}
	else
	{
		UE_LOG(LogNNERuntimeOpenVINO, Warning, TEXT("No GPU device found, INNERuntimeGPU will be unavailable."));
	}
#endif
}

void FNNERuntimeOpenVINO::ShutdownModule()
{
	if (OVCore)
	{
		ov_core_free(OVCore);
	}

#ifdef OPENVINO_GPU_PLUGIN
	// NNE runtime ORT Gpu shutdown
	if (NNERuntimeOpenVINOGpu.IsValid())
	{
		TWeakInterfacePtr<INNERuntime> RuntimeGPUInterface(NNERuntimeOpenVINOGpu.Get());

		UE::NNE::UnregisterRuntime(RuntimeGPUInterface);
		NNERuntimeOpenVINOGpu->RemoveFromRoot();
		NNERuntimeOpenVINOGpu.Reset();
	}
#endif

#ifdef OPENVINO_NPU_PLUGIN
	// NNE runtime ORT Npu shutdown
	if (NNERuntimeOpenVINOCpu.IsValid())
	{
		TWeakInterfacePtr<INNERuntime> RuntimeCPUInterface(NNERuntimeOpenVINOCpu.Get());

		UE::NNE::UnregisterRuntime(RuntimeCPUInterface);
		NNERuntimeOpenVINOCpu->RemoveFromRoot();
		NNERuntimeOpenVINOCpu.Reset();
	}
#endif

#ifdef OPENVINO_CPU_PLUGIN
	// NNE runtime ORT Cpu shutdown
	if (NNERuntimeOpenVINONpu.IsValid())
	{
		TWeakInterfacePtr<INNERuntime> RuntimeNPUInterface(NNERuntimeOpenVINONpu.Get());

		UE::NNE::UnregisterRuntime(RuntimeNPUInterface);
		NNERuntimeOpenVINOCpu->RemoveFromRoot();
		NNERuntimeOpenVINOCpu.Reset();
	}
#endif

	UnloadDLL();

	UE_LOG(LogNNERuntimeOpenVINO, Display, TEXT("Unloaded NNERuntimeOpenVINO"));
}

FName FNNERuntimeOpenVINO::ModuleName()
{
	return TEXT("NNERuntimeOpenVINO");
}

bool FNNERuntimeOpenVINO::LoadDLL()
{
	const FString PluginDir(IPluginManager::Get().FindPlugin(TEXT("NNERuntimeOpenVINO"))->GetBaseDir());
	IFileManager& FileManager = IFileManager::Get();
	const FString OpenVINODLLPath(FileManager.ConvertToAbsolutePathForExternalAppForRead(*FPaths::Combine(PluginDir, OPENVINO_PATH)));
	const FString TBBDLLPath(FileManager.ConvertToAbsolutePathForExternalAppForRead(*FPaths::Combine(PluginDir, OPENVINO_TBB_PATH)));
	const FString OpenVINODLLFilePath(FPaths::Combine(OpenVINODLLPath, OPENVINO_DLL_NAME));
	FPlatformProcess::AddDllDirectory(*OpenVINODLLPath);
	FPlatformProcess::AddDllDirectory(*TBBDLLPath);
	OpenVINODLL = FPlatformProcess::GetDllHandle(*OpenVINODLLFilePath);
	return OpenVINODLL != nullptr;
}

void FNNERuntimeOpenVINO::UnloadDLL()
{
	if (OpenVINODLL)
	{
		FPlatformProcess::FreeDllHandle(OpenVINODLL);
	}
}

void FNNERuntimeOpenVINO::LogDevices()
{
	ov_available_devices_t AvailableDevices{};
	if (ov_core_get_available_devices(OVCore, &AvailableDevices))
	{
		UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Failed to fetch OpenVINO devices."));
	}

	for (size_t i = 0; i < AvailableDevices.size; ++i)
	{
		UE_LOG(LogNNERuntimeOpenVINO, Display, TEXT("OpenVINO found device [%s]"), ANSI_TO_TCHAR(AvailableDevices.devices[i]));
	}

	ov_available_devices_free(&AvailableDevices);
}
