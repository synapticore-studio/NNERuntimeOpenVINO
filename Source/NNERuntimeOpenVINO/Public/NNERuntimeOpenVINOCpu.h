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

#pragma once

#include "CoreMinimal.h"

#include "NNERuntime.h"
#include "NNERuntimeCPU.h"
#include "NNEStatus.h"
#include "NNERuntimeRunSync.h"
#include "NNETypes.h"
#include "NNEModelData.h"
#include "UObject/Class.h"
#include "UObject/Object.h"
#include "UObject/UObjectBaseUtility.h"

#include "NNERuntimeOpenVINOModule.h"

THIRD_PARTY_INCLUDES_START
#include "openvino/c/ov_model.h"
THIRD_PARTY_INCLUDES_END

#include "NNERuntimeOpenVINOCpu.generated.h"

class FModelInstanceOpenVINOCpu : public UE::NNE::IModelInstanceCPU
{
public:
	FModelInstanceOpenVINOCpu() = default;
	virtual ~FModelInstanceOpenVINOCpu();

	bool Init(TSharedRef<UE::NNE::FSharedModelData> ModelData);

	virtual TConstArrayView<UE::NNE::FTensorDesc> GetInputTensorDescs() const override;
	virtual TConstArrayView<UE::NNE::FTensorDesc> GetOutputTensorDescs() const override;
	virtual TConstArrayView<UE::NNE::FTensorShape> GetInputTensorShapes() const override;
	virtual TConstArrayView<UE::NNE::FTensorShape> GetOutputTensorShapes() const override;

	virtual UE::NNE::EResultStatus SetInputTensorShapes(TConstArrayView<UE::NNE::FTensorShape> InInputShapes) override;

	virtual UE::NNE::EResultStatus RunSync(TConstArrayView<UE::NNE::FTensorBindingCPU> InInputTensors, TConstArrayView<UE::NNE::FTensorBindingCPU> InOutputTensors) override;

private:
	TArray<UE::NNE::FTensorShape> InputTensorShapes;
	TArray<UE::NNE::FTensorShape> OutputTensorShapes;
	TArray<UE::NNE::FTensorDesc> InputSymbolicTensors;
	TArray<UE::NNE::FTensorDesc> OutputSymbolicTensors;

	ov_model_t* Model = nullptr;
	ov_compiled_model_t* CompiledModel = nullptr;
};

class FModelOpenVINOCpu : public UE::NNE::IModelCPU
{
public:
	FModelOpenVINOCpu(TSharedRef<UE::NNE::FSharedModelData> InModelData);
	virtual ~FModelOpenVINOCpu() = default;

	virtual TSharedPtr<UE::NNE::IModelInstanceCPU> CreateModelInstanceCPU() override;

private:
	TSharedRef<UE::NNE::FSharedModelData> ModelData;
};

UCLASS()
class UNNERuntimeOpenVINOCpu : public UObject, public INNERuntime, public INNERuntimeCPU
{
	GENERATED_BODY()

private:

public:
	static FGuid GUID;
	static int32 Version;

	virtual ~UNNERuntimeOpenVINOCpu() = default;

	virtual FString GetRuntimeName() const override;

	virtual INNERuntime::ECanCreateModelDataStatus CanCreateModelData(const FString& FileType, TConstArrayView64<uint8> FileData, const TMap<FString, TConstArrayView64<uint8>>& AdditionalFileData, const FGuid& FileId, const ITargetPlatform* TargetPlatform) const override;
	virtual TSharedPtr<UE::NNE::FSharedModelData> CreateModelData(const FString& FileType, TConstArrayView64<uint8> FileData, const TMap<FString, TConstArrayView64<uint8>>& AdditionalFileData, const FGuid& FileId, const ITargetPlatform* TargetPlatform) override;
	virtual FString GetModelDataIdentifier(const FString& FileType, TConstArrayView64<uint8> FileData, const TMap<FString, TConstArrayView64<uint8>>& AdditionalFileData, const FGuid& FileId, const ITargetPlatform* TargetPlatform) const override;

	virtual INNERuntimeCPU::ECanCreateModelCPUStatus CanCreateModelCPU(const TObjectPtr<UNNEModelData> ModelData) const override;
	virtual TSharedPtr<UE::NNE::IModelCPU> CreateModelCPU(const TObjectPtr<UNNEModelData> ModelData) override;
};
