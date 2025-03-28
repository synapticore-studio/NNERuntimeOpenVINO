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

#include "NNERuntimeOpenVINOCpu.h"

#include "Memory/SharedBuffer.h"
#include "NNE.h"
#include "NNEModelData.h"
#include "NNERuntimeOpenVINOCommon.h"

#include "openvino/c/ov_prepostprocess.h"
#include "openvino/c/ov_tensor.h"

FGuid UNNERuntimeOpenVINOCpu::GUID = FGuid((int32)'O', (int32)'V', (int32)'_', (int32)'C');
int32 UNNERuntimeOpenVINOCpu::Version = 0x00000001;

FModelInstanceOpenVINOCpu::~FModelInstanceOpenVINOCpu()
{
	if (CompiledModel)
	{
		ov_compiled_model_free(CompiledModel);
	}

	if (Model)
	{
		ov_model_free(Model);
	}
}

bool FModelInstanceOpenVINOCpu::Init(TSharedRef<UE::NNE::FSharedModelData> ModelData)
{
	FMemoryReaderView MemoryReader(ModelData->GetView());

	int64 Offset = MemoryReader.Tell();

	TConstArrayView64<uint8> FileData(ModelData->GetView().GetData() + Offset, ModelData->GetView().NumBytes() - Offset);

	// Load the model into OpenVINO
	FNNERuntimeOpenVINO* OVModule = FModuleManager::GetModulePtr<FNNERuntimeOpenVINO>(FNNERuntimeOpenVINO::ModuleName());
	if (!OVModule)
	{
		UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Couldn't find NNERuntimeOpenVINO module."));
		return false;
	}

	ov_core_t& OVCore = OVModule->OpenVINOInstance();
	if (ov_core_read_model_from_memory_buffer(&OVCore, (const char*)FileData.GetData(), FileData.Num(), NULL, &Model))
	{
		UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Failed to read the model."));
		return false;
	}

	// TODO: The only real difference between the devices is the selection here.
	//	Check with Matt Curfman to see if there's anything else here that needs to be done to optimize or use on different devices.
	const FString DeviceName(TEXT("CPU"));
	if (!SupportsDevice(OVCore, DeviceName))
	{
		ov_model_free(Model);
		Model = nullptr;
		UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("[%s] device not found."), *DeviceName);
		return false;
	}

	if (ov_core_compile_model(&OVCore, Model, TCHAR_TO_ANSI(*DeviceName), 0, &CompiledModel))
	{
		ov_model_free(Model);
		Model = nullptr;
		UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Failed to compile the model."));
		return false;
	}

	return true;
}

TConstArrayView<UE::NNE::FTensorDesc> FModelInstanceOpenVINOCpu::GetInputTensorDescs() const
{
	return TConstArrayView<UE::NNE::FTensorDesc>();
}

TConstArrayView<UE::NNE::FTensorDesc> FModelInstanceOpenVINOCpu::GetOutputTensorDescs() const
{
	return TConstArrayView<UE::NNE::FTensorDesc>();
}

TConstArrayView<UE::NNE::FTensorShape> FModelInstanceOpenVINOCpu::GetInputTensorShapes() const
{
	return TConstArrayView<UE::NNE::FTensorShape>();
}

TConstArrayView<UE::NNE::FTensorShape> FModelInstanceOpenVINOCpu::GetOutputTensorShapes() const
{
	return TConstArrayView<UE::NNE::FTensorShape>();
}

UE::NNE::EResultStatus FModelInstanceOpenVINOCpu::SetInputTensorShapes(TConstArrayView<UE::NNE::FTensorShape> InInputShapes)
{
	if (!Model)
	{
		UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Invalid model."));
		return UE::NNE::EResultStatus::Fail;
	}

	InputTensorShapes = InInputShapes;

	return UE::NNE::EResultStatus::Ok;
}

UE::NNE::EResultStatus FModelInstanceOpenVINOCpu::RunSync(TConstArrayView<UE::NNE::FTensorBindingCPU> InInputTensors, TConstArrayView<UE::NNE::FTensorBindingCPU> InOutputTensors)
{
	if (InInputTensors.IsEmpty() || InOutputTensors.IsEmpty())
	{
		UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Input/Output tensors are not set up properly."));
		return UE::NNE::EResultStatus::Fail;
	}

	if (!Model || !CompiledModel)
	{
		UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Invalid model."));
		return UE::NNE::EResultStatus::Fail;
	}

	FNNERuntimeOpenVINO* OVModule = FModuleManager::GetModulePtr<FNNERuntimeOpenVINO>(FNNERuntimeOpenVINO::ModuleName());
	if (!OVModule)
	{
		UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Couldn't find NNERuntimeOpenVINO module."));
		return UE::NNE::EResultStatus::Fail;
	}

	ov_core_t& OVInstance = OVModule->OpenVINOInstance();

	ov_infer_request_t* InferRequest = nullptr;
	if (ov_compiled_model_create_infer_request(CompiledModel, &InferRequest))
	{
		ov_compiled_model_free(CompiledModel);
		ov_model_free(Model);
		CompiledModel = nullptr;
		Model = nullptr;
		UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Failed to compile the model."));
		return UE::NNE::EResultStatus::Fail;
	}

	TArray<ov_output_const_port_t*> InputPorts;
	TArray<ov_shape_t> InputShapes;
	TArray<ov_tensor_t*> InputTensors;

	for (int32 i = 0; i < InInputTensors.Num(); ++i)
	{
		ov_output_const_port_t*& InputPort = InputPorts.AddZeroed_GetRef();
		if (ov_model_const_input_by_index(Model, i, &InputPort))
		{
			ReleasePorts(InputPorts);
			ReleaseShapes(InputShapes);
			ReleaseTensors(InputTensors);
			ov_infer_request_free(InferRequest);
			ov_compiled_model_free(CompiledModel);
			ov_model_free(Model);
			CompiledModel = nullptr;
			Model = nullptr;
			UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Failed to get input config."));
			return UE::NNE::EResultStatus::Fail;
		}

		ov_shape_t& InputShape = InputShapes.AddZeroed_GetRef();
		if (ov_const_port_get_shape(InputPort, &InputShape))
		{
			ReleasePorts(InputPorts);
			ReleaseShapes(InputShapes);
			ReleaseTensors(InputTensors);
			ov_infer_request_free(InferRequest);
			ov_compiled_model_free(CompiledModel);
			ov_model_free(Model);
			CompiledModel = nullptr;
			Model = nullptr;
			UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Failed to get input shape."));
			return UE::NNE::EResultStatus::Fail;
		}

		ov_element_type_e InputType{};
		if (ov_port_get_element_type(InputPort, &InputType))
		{
			ReleasePorts(InputPorts);
			ReleaseShapes(InputShapes);
			ReleaseTensors(InputTensors);
			ov_infer_request_free(InferRequest);
			ov_compiled_model_free(CompiledModel);
			ov_model_free(Model);
			CompiledModel = nullptr;
			Model = nullptr;
			UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Failed to get input type."));
			return UE::NNE::EResultStatus::Fail;
		}

		void* InputData = InInputTensors[i].Data;
		ov_tensor_t*& InputTensor = InputTensors.AddZeroed_GetRef();
		if (ov_tensor_create_from_host_ptr(InputType, InputShape, InputData, &InputTensor))
		{
			ReleasePorts(InputPorts);
			ReleaseShapes(InputShapes);
			ReleaseTensors(InputTensors);
			ov_infer_request_free(InferRequest);
			ov_compiled_model_free(CompiledModel);
			ov_model_free(Model);
			CompiledModel = nullptr;
			Model = nullptr;
			UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Failed to create tensor from input data."));
			return UE::NNE::EResultStatus::Fail;
		}

		if (ov_infer_request_set_input_tensor_by_index(InferRequest, i, InputTensor))
		{
			ReleasePorts(InputPorts);
			ReleaseShapes(InputShapes);
			ReleaseTensors(InputTensors);
			ov_infer_request_free(InferRequest);
			ov_compiled_model_free(CompiledModel);
			ov_model_free(Model);
			CompiledModel = nullptr;
			Model = nullptr;
			UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Failed to set input tensor for infer request."));
			return UE::NNE::EResultStatus::Fail;
		}
	}

	TArray<ov_output_const_port_t*> OutputPorts;
	TArray<ov_shape_t> OutputShapes;
	TArray<ov_tensor_t*> OutputTensors;

	for (int32 i = 0; i < InOutputTensors.Num(); ++i)
	{
		ov_output_const_port_t*& OutputPort = OutputPorts.AddZeroed_GetRef();
		if (ov_model_const_output_by_index(Model, i, &OutputPort))
		{
			ReleasePorts(InputPorts);
			ReleaseShapes(InputShapes);
			ReleaseTensors(InputTensors);
			ReleasePorts(OutputPorts);
			ReleaseShapes(OutputShapes);
			ReleaseTensors(OutputTensors);
			ov_infer_request_free(InferRequest);
			ov_compiled_model_free(CompiledModel);
			ov_model_free(Model);
			CompiledModel = nullptr;
			Model = nullptr;
			UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Failed to get output config."));
			return UE::NNE::EResultStatus::Fail;
		}

		ov_shape_t& OutputShape = OutputShapes.AddZeroed_GetRef();
		if (ov_const_port_get_shape(OutputPort, &OutputShape))
		{
			ReleasePorts(InputPorts);
			ReleaseShapes(InputShapes);
			ReleaseTensors(InputTensors);
			ReleasePorts(OutputPorts);
			ReleaseShapes(OutputShapes);
			ReleaseTensors(OutputTensors);
			ov_infer_request_free(InferRequest);
			ov_compiled_model_free(CompiledModel);
			ov_model_free(Model);
			CompiledModel = nullptr;
			Model = nullptr;
			UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Failed to get output shape."));
			return UE::NNE::EResultStatus::Fail;
		}

		// Get the the type of input
		ov_element_type_e OutputType{};
		if (ov_port_get_element_type(OutputPort, &OutputType))
		{
			ReleasePorts(InputPorts);
			ReleaseShapes(InputShapes);
			ReleaseTensors(InputTensors);
			ReleasePorts(OutputPorts);
			ReleaseShapes(OutputShapes);
			ReleaseTensors(OutputTensors);
			ov_infer_request_free(InferRequest);
			ov_compiled_model_free(CompiledModel);
			ov_model_free(Model);
			CompiledModel = nullptr;
			Model = nullptr;
			UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Failed to get output type."));
			return UE::NNE::EResultStatus::Fail;
		}

		void* OutputData = InOutputTensors[i].Data;
		ov_tensor_t*& OutputTensor = OutputTensors.AddZeroed_GetRef();
		if (ov_tensor_create_from_host_ptr(OutputType, OutputShape, OutputData, &OutputTensor))
		{
			ReleasePorts(InputPorts);
			ReleaseShapes(InputShapes);
			ReleaseTensors(InputTensors);
			ReleasePorts(OutputPorts);
			ReleaseShapes(OutputShapes);
			ReleaseTensors(OutputTensors);
			ov_infer_request_free(InferRequest);
			ov_compiled_model_free(CompiledModel);
			ov_model_free(Model);
			CompiledModel = nullptr;
			Model = nullptr;
			UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Failed to create tensor from output data."));
			return UE::NNE::EResultStatus::Fail;
		}

		if (ov_infer_request_set_output_tensor_by_index(InferRequest, i, OutputTensor))
		{
			ReleasePorts(InputPorts);
			ReleaseShapes(InputShapes);
			ReleaseTensors(InputTensors);
			ReleasePorts(OutputPorts);
			ReleaseShapes(OutputShapes);
			ReleaseTensors(OutputTensors);
			ov_infer_request_free(InferRequest);
			ov_compiled_model_free(CompiledModel);
			ov_model_free(Model);
			CompiledModel = nullptr;
			Model = nullptr;
			UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Failed to set output tensor for infer request."));
			return UE::NNE::EResultStatus::Fail;
		}
	}

	if (ov_infer_request_infer(InferRequest))
	{
		ReleasePorts(InputPorts);
		ReleaseShapes(InputShapes);
		ReleaseTensors(InputTensors);
		ReleasePorts(OutputPorts);
		ReleaseShapes(OutputShapes);
		ReleaseTensors(OutputTensors);
		ov_infer_request_free(InferRequest);
		ov_compiled_model_free(CompiledModel);
		ov_model_free(Model);
		CompiledModel = nullptr;
		Model = nullptr;
		UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Failed to execute infer request."));
		return UE::NNE::EResultStatus::Fail;
	}

	ReleasePorts(InputPorts);
	ReleaseShapes(InputShapes);
	ReleaseTensors(InputTensors);
	ReleasePorts(OutputPorts);
	ReleaseShapes(OutputShapes);
	ReleaseTensors(OutputTensors);
	ov_infer_request_free(InferRequest);
	ov_compiled_model_free(CompiledModel);
	ov_model_free(Model);
	CompiledModel = nullptr;
	Model = nullptr;

	return UE::NNE::EResultStatus::Ok;
}

FModelOpenVINOCpu::FModelOpenVINOCpu(TSharedRef<UE::NNE::FSharedModelData> InModelData)
	: ModelData(InModelData)
{
}

TSharedPtr<UE::NNE::IModelInstanceCPU> FModelOpenVINOCpu::CreateModelInstanceCPU()
{
	TSharedPtr<FModelInstanceOpenVINOCpu> ModelInstance = MakeShared<FModelInstanceOpenVINOCpu>();
	if (!ModelInstance->Init(ModelData))
	{
		UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Failed to initialize the model instance."));
		return {};
	}

	return ModelInstance;
}

FString UNNERuntimeOpenVINOCpu::GetRuntimeName() const
{
	return TEXT("NNERuntimeOpenVINOCpu");
}

INNERuntime::ECanCreateModelDataStatus UNNERuntimeOpenVINOCpu::CanCreateModelData(const FString& FileType, TConstArrayView64<uint8> FileData, const TMap<FString, TConstArrayView64<uint8>>& AdditionalFileData, const FGuid& FileId, const ITargetPlatform* TargetPlatform) const
{
	return (!FileData.IsEmpty() && IsFileSupported(FileType)) ? ECanCreateModelDataStatus::Ok : ECanCreateModelDataStatus::FailFileIdNotSupported;
}

TSharedPtr<UE::NNE::FSharedModelData> UNNERuntimeOpenVINOCpu::CreateModelData(const FString& FileType, TConstArrayView64<uint8> FileData, const TMap<FString, TConstArrayView64<uint8>>& AdditionalFileData, const FGuid& FileId, const ITargetPlatform* TargetPlatform)
{
	if (CanCreateModelData(FileType, FileData, AdditionalFileData, FileId, TargetPlatform) != ECanCreateModelDataStatus::Ok)
	{
		UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Cannot create the CPU model data with id %s (Filetype: %s)"), *FileId.ToString(EGuidFormats::Digits).ToLower(), *FileType);
		return {};
	}

	FSharedBuffer SharedBuffer(FSharedBuffer::Clone(FileData.GetData(), FileData.NumBytes()));
	TSharedPtr<UE::NNE::FSharedModelData> SharedData(MakeShared<UE::NNE::FSharedModelData>(SharedBuffer, 0));
	return SharedData;
}

FString UNNERuntimeOpenVINOCpu::GetModelDataIdentifier(const FString& FileType, TConstArrayView64<uint8> FileData, const TMap<FString, TConstArrayView64<uint8>>& AdditionalFileData, const FGuid& FileId, const ITargetPlatform* TargetPlatform) const
{
	return FileId.ToString(EGuidFormats::Digits) + "-" + UNNERuntimeOpenVINOCpu::GUID.ToString(EGuidFormats::Digits) + "-" + FString::FromInt(UNNERuntimeOpenVINOCpu::Version);
}

INNERuntimeCPU::ECanCreateModelCPUStatus UNNERuntimeOpenVINOCpu::CanCreateModelCPU(const TObjectPtr<UNNEModelData> ModelData) const
{
	check(ModelData != nullptr);

	const TSharedPtr<UE::NNE::FSharedModelData> SharedData = ModelData->GetModelData(GetRuntimeName());

	if (!SharedData.IsValid())
	{
		return ECanCreateModelCPUStatus::Fail;
	}

	TConstArrayView64<uint8> Data = SharedData->GetView();

	if (Data.Num() == 0)
	{
		return ECanCreateModelCPUStatus::Fail;
	}

	return ECanCreateModelCPUStatus::Ok;
}

TSharedPtr<UE::NNE::IModelCPU> UNNERuntimeOpenVINOCpu::CreateModelCPU(const TObjectPtr<UNNEModelData> ModelData)
{
	check(ModelData != nullptr);

	if (CanCreateModelCPU(ModelData) != ECanCreateModelCPUStatus::Ok)
	{
		UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Cannot create a CPU model from the model data with id %s"), *ModelData->GetFileId().ToString(EGuidFormats::Digits));
		return TSharedPtr<UE::NNE::IModelCPU>();
	}

	TSharedRef<UE::NNE::FSharedModelData> SharedData = ModelData->GetModelData(GetRuntimeName()).ToSharedRef();

	return MakeShared<FModelOpenVINOCpu>(SharedData);
}
