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

#include "NNERuntimeOpenVINONpu.h"

#include "Memory/SharedBuffer.h"
#include "NNE.h"
#include "NNEModelData.h"
#include "NNERuntimeOpenVINOCommon.h"

THIRD_PARTY_INCLUDES_START
#include "openvino/c/ov_prepostprocess.h"
#include "openvino/c/ov_tensor.h"
THIRD_PARTY_INCLUDES_END

FGuid UNNERuntimeOpenVINONpu::GUID = FGuid((int32)'O', (int32)'V', (int32)'_', (int32)'N');
int32 UNNERuntimeOpenVINONpu::Version = 0x00000001;

FModelInstanceOpenVINONpu::~FModelInstanceOpenVINONpu()
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

bool FModelInstanceOpenVINONpu::Init(TSharedRef<UE::NNE::FSharedModelData> ModelData)
{
	const FString DeviceName(TEXT("NPU"));
	if (!InitModelInstance(ModelData, Model, CompiledModel, DeviceName))
	{
		return false;
	}

	bool bResult = true;

	if (!InitModelTensorDescs(InputSymbolicTensors, OutputSymbolicTensors, Model))
	{
		bResult = false;
		ov_compiled_model_free(CompiledModel);
		ov_model_free(Model);
		CompiledModel = nullptr;
		Model = nullptr;
	}

	return bResult;
}

TConstArrayView<UE::NNE::FTensorDesc> FModelInstanceOpenVINONpu::GetInputTensorDescs() const
{
	return InputSymbolicTensors;
}

TConstArrayView<UE::NNE::FTensorDesc> FModelInstanceOpenVINONpu::GetOutputTensorDescs() const
{
	return OutputSymbolicTensors;
}

TConstArrayView<UE::NNE::FTensorShape> FModelInstanceOpenVINONpu::GetInputTensorShapes() const
{
	return InputTensorShapes;
}

TConstArrayView<UE::NNE::FTensorShape> FModelInstanceOpenVINONpu::GetOutputTensorShapes() const
{
	return OutputTensorShapes;
}

UE::NNE::EResultStatus FModelInstanceOpenVINONpu::SetInputTensorShapes(TConstArrayView<UE::NNE::FTensorShape> InInputShapes)
{
	if (!Model)
	{
		UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Invalid model."));
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

UE::NNE::EResultStatus FModelInstanceOpenVINONpu::RunSync(TConstArrayView<UE::NNE::FTensorBindingCPU> InInputTensors, TConstArrayView<UE::NNE::FTensorBindingCPU> InOutputTensors)
{
	return ModelInfer(InInputTensors, InOutputTensors, Model, CompiledModel);
}

FModelOpenVINONpu::FModelOpenVINONpu(TSharedRef<UE::NNE::FSharedModelData> InModelData)
	: ModelData(InModelData)
{
}

TSharedPtr<UE::NNE::IModelInstanceNPU> FModelOpenVINONpu::CreateModelInstanceNPU()
{
	TSharedPtr<FModelInstanceOpenVINONpu> ModelInstance = MakeShared<FModelInstanceOpenVINONpu>();
	if (!ModelInstance->Init(ModelData))
	{
		UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Failed to initialize the model instance."));
		return {};
	}

	return ModelInstance;
}

FString UNNERuntimeOpenVINONpu::GetRuntimeName() const
{
	return TEXT("NNERuntimeOpenVINONpu");
}

INNERuntime::ECanCreateModelDataStatus UNNERuntimeOpenVINONpu::CanCreateModelData(const FString& FileType, TConstArrayView64<uint8> FileData, const TMap<FString, TConstArrayView64<uint8>>& AdditionalFileData, const FGuid& FileId, const ITargetPlatform* TargetPlatform) const
{
	return (!FileData.IsEmpty() && IsFileSupported(FileType)) ? ECanCreateModelDataStatus::Ok : ECanCreateModelDataStatus::FailFileIdNotSupported;
}

TSharedPtr<UE::NNE::FSharedModelData> UNNERuntimeOpenVINONpu::CreateModelData(const FString& FileType, TConstArrayView64<uint8> FileData, const TMap<FString, TConstArrayView64<uint8>>& AdditionalFileData, const FGuid& FileId, const ITargetPlatform* TargetPlatform)
{
	if (CanCreateModelData(FileType, FileData, AdditionalFileData, FileId, TargetPlatform) != ECanCreateModelDataStatus::Ok)
	{
		UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Cannot create the NPU model data with id %s (Filetype: %s)"), *FileId.ToString(EGuidFormats::Digits).ToLower(), *FileType);
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

FString UNNERuntimeOpenVINONpu::GetModelDataIdentifier(const FString& FileType, TConstArrayView64<uint8> FileData, const TMap<FString, TConstArrayView64<uint8>>& AdditionalFileData, const FGuid& FileId, const ITargetPlatform* TargetPlatform) const
{
	return FileId.ToString(EGuidFormats::Digits) + "-" + UNNERuntimeOpenVINONpu::GUID.ToString(EGuidFormats::Digits) + "-" + FString::FromInt(UNNERuntimeOpenVINONpu::Version);
}

INNERuntimeNPU::ECanCreateModelNPUStatus UNNERuntimeOpenVINONpu::CanCreateModelNPU(const TObjectPtr<UNNEModelData> ModelData) const
{
	check(ModelData != nullptr);

	const TSharedPtr<UE::NNE::FSharedModelData> SharedData = ModelData->GetModelData(GetRuntimeName());

	if (!SharedData.IsValid())
	{
		return ECanCreateModelNPUStatus::Fail;
	}

	TConstArrayView64<uint8> Data = SharedData->GetView();

	if (Data.Num() == 0)
	{
		return ECanCreateModelNPUStatus::Fail;
	}

	return ECanCreateModelNPUStatus::Ok;
}

TSharedPtr<UE::NNE::IModelNPU> UNNERuntimeOpenVINONpu::CreateModelNPU(const TObjectPtr<UNNEModelData> ModelData)
{
	check(ModelData != nullptr);

	if (CanCreateModelNPU(ModelData) != ECanCreateModelNPUStatus::Ok)
	{
		UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Cannot create a NPU model from the model data with id %s"), *ModelData->GetFileId().ToString(EGuidFormats::Digits));
		return TSharedPtr<UE::NNE::IModelNPU>();
	}

	TSharedRef<UE::NNE::FSharedModelData> SharedData = ModelData->GetModelData(GetRuntimeName()).ToSharedRef();
	return MakeShared<FModelOpenVINONpu>(SharedData);
}
