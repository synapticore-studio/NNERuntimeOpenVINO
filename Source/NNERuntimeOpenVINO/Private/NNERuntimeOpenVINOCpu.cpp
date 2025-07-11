/*******************************************************************************
* Copyright (C) 2025 Intel Corporation
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom
* the Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
* OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
* OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
* OR OTHER DEALINGS IN THE SOFTWARE.
*
* SPDX-License-Identifier: MIT
******************************************************************************/

#include "NNERuntimeOpenVINOCpu.h"

#include "Memory/SharedBuffer.h"
#include "NNE.h"
#include "NNEModelData.h"
#include "NNERuntimeOpenVINOCommon.h"
#include "Serialization/MemoryWriter.h"

THIRD_PARTY_INCLUDES_START
#include "openvino/c/ov_prepostprocess.h"
#include "openvino/c/ov_tensor.h"
THIRD_PARTY_INCLUDES_END

FGuid UNNERuntimeOpenVINOCpu::GUID = FGuid((int32)'O', (int32)'V', (int32)'_', (int32)'C');
int32 UNNERuntimeOpenVINOCpu::Version = 0x00000001;

FModelInstanceOpenVINOCpu::~FModelInstanceOpenVINOCpu()
{
	if (CompiledModel)
	{
		ov_compiled_model_free(CompiledModel);
	}
}

bool FModelInstanceOpenVINOCpu::Init(TSharedRef<UE::NNE::FSharedModelData> ModelData)
{
	const FString DeviceName(TEXT("CPU"));
	if (!InitModelInstance(ModelData, CompiledModel, DeviceName))
	{
		return false;
	}

	bool bResult = true;

	if (!InitModelTensorDescs(InputSymbolicTensors, OutputSymbolicTensors, CompiledModel))
	{
		bResult = false;
		ov_compiled_model_free(CompiledModel);
		CompiledModel = nullptr;
	}

	return bResult;
}

TConstArrayView<UE::NNE::FTensorDesc> FModelInstanceOpenVINOCpu::GetInputTensorDescs() const
{
	return InputSymbolicTensors;
}

TConstArrayView<UE::NNE::FTensorDesc> FModelInstanceOpenVINOCpu::GetOutputTensorDescs() const
{
	return OutputSymbolicTensors;
}

TConstArrayView<UE::NNE::FTensorShape> FModelInstanceOpenVINOCpu::GetInputTensorShapes() const
{
	return InputTensorShapes;
}

TConstArrayView<UE::NNE::FTensorShape> FModelInstanceOpenVINOCpu::GetOutputTensorShapes() const
{
	return OutputTensorShapes;
}

UE::NNE::EResultStatus FModelInstanceOpenVINOCpu::SetInputTensorShapes(TConstArrayView<UE::NNE::FTensorShape> InInputShapes)
{
	if (!CompiledModel)
	{
		UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Invalid compiled model."));
		return UE::NNE::EResultStatus::Fail;
	}

	if (InInputShapes.Num() != InputSymbolicTensors.Num())
	{
		UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Input shape sizes don't match."));
		return UE::NNE::EResultStatus::Fail;
	}

	for (int32 i = 0; i < InInputShapes.Num(); ++i)
	{
		const UE::NNE::FTensorDesc SymbolicDesc = InputSymbolicTensors[i];
		if (!InInputShapes[i].IsCompatibleWith(SymbolicDesc.GetShape()))
		{
			UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Input shape tensor [%s] doesn't match input tensor [%d]."), *SymbolicDesc.GetName(), i);
			return UE::NNE::EResultStatus::Fail;
		}
	}

	InputTensorShapes = InInputShapes;
	return UE::NNE::EResultStatus::Ok;
}

UE::NNE::EResultStatus FModelInstanceOpenVINOCpu::RunSync(TConstArrayView<UE::NNE::FTensorBindingCPU> InInputTensors, TConstArrayView<UE::NNE::FTensorBindingCPU> InOutputTensors)
{
	return ModelInfer(InInputTensors, InOutputTensors, CompiledModel);
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

	bool bHasWeights = FileType.Compare(TEXT("xml"), ESearchCase::IgnoreCase) == 0;

	TArray64<uint8> WrappedFileData;
	FMemoryWriter64 MemoryWriter(WrappedFileData);
	MemoryWriter << bHasWeights;
	MemoryWriter.Serialize((void*)FileData.GetData(), FileData.NumBytes());

	FSharedBuffer SharedBuffer(FSharedBuffer::Clone(WrappedFileData.GetData(), WrappedFileData.NumBytes()));
	
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
