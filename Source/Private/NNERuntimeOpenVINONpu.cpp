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

#include "openvino/c/ov_prepostprocess.h"
#include "openvino/c/ov_tensor.h"

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
	return InitModelInstance(ModelData, Model, CompiledModel, DeviceName);
}

TConstArrayView<UE::NNE::FTensorDesc> FModelInstanceOpenVINONpu::GetInputTensorDescs() const
{
	return TConstArrayView<UE::NNE::FTensorDesc>();
}

TConstArrayView<UE::NNE::FTensorDesc> FModelInstanceOpenVINONpu::GetOutputTensorDescs() const
{
	return TConstArrayView<UE::NNE::FTensorDesc>();
}

TConstArrayView<UE::NNE::FTensorShape> FModelInstanceOpenVINONpu::GetInputTensorShapes() const
{
	return TConstArrayView<UE::NNE::FTensorShape>();
}

TConstArrayView<UE::NNE::FTensorShape> FModelInstanceOpenVINONpu::GetOutputTensorShapes() const
{
	return TConstArrayView<UE::NNE::FTensorShape>();
}

UE::NNE::EResultStatus FModelInstanceOpenVINONpu::SetInputTensorShapes(TConstArrayView<UE::NNE::FTensorShape> InInputShapes)
{
	if (!Model)
	{
		UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Invalid model."));
		return UE::NNE::EResultStatus::Fail;
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

	FSharedBuffer SharedBuffer(FSharedBuffer::Clone(FileData.GetData(), FileData.NumBytes()));
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
