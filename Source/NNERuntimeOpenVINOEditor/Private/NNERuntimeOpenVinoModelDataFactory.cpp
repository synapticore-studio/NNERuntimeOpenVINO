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

#include "NNERuntimeOpenVINOModelDataFactory.h"

#include "CoreMinimal.h"
#include "Editor.h"
#include "NNE.h"
#include "NNEModelData.h"
#include "Serialization/MemoryWriter.h"
#include "Subsystems/ImportSubsystem.h"

#include "NNERuntimeOpenVINOEditorModule.h"

bool IsFileSupported(const FString& FileType)
{
	/*
	* *.onnx -> NNERuntimeORT handles import
	* *.pb, *.pdmodel, *.tflite -> Must be converted externally using Python. There is no C/C++ support for this.
	* *.xml -> OpenVINO IR is the only one we need to handle.
	*/
	return FileType.Compare(TEXT("xml"), ESearchCase::IgnoreCase) == 0;
}

UNNERuntimeOpenVINOModelDataFactory::UNNERuntimeOpenVINOModelDataFactory(const FObjectInitializer& ObjectInitializer) : UFactory(ObjectInitializer)
{
	bCreateNew = false;
	bEditorImport = true;
	SupportedClass = UNNEModelData::StaticClass();
	ImportPriority = DefaultImportPriority;
	/*
	* *.onnx -> NNERuntimeORT handles import
	* *.pb, *.pdmodel, *.tflite -> Must be converted externally using Python. There is no C/C++ support for this.
	* *.xml -> OpenVINO IR is the only one we need to handle.
	*/
	Formats.Add("xml;OpenVINO IR Format");
}

UObject* UNNERuntimeOpenVINOModelDataFactory::FactoryCreateFile(UClass* InClass, UObject* InParent, FName InName, EObjectFlags Flags, const FString& Filename, const TCHAR* Parms, FFeedbackContext* Warn, bool& bOutOperationCanceled)
{
	const FString FileExtension(FPaths::GetExtension(Filename));
	GEditor->GetEditorSubsystem<UImportSubsystem>()->BroadcastAssetPreImport(this, InClass, InParent, InName, *FileExtension);

	if (!IsFileSupported(FileExtension))
	{
		UE_LOG(LogNNERuntimeOpenVINOEditor, Error, TEXT("Unsupported file '%s'"), *Filename);
		GEditor->GetEditorSubsystem<UImportSubsystem>()->BroadcastAssetPostImport(this, nullptr);
		return nullptr;
	}

	TArray64<uint8> FileData;
	if (!FFileHelper::LoadFileToArray(FileData, *Filename))
	{
		UE_LOG(LogNNERuntimeOpenVINOEditor, Error, TEXT("Failed to load file '%s'"), *Filename);
		GEditor->GetEditorSubsystem<UImportSubsystem>()->BroadcastAssetPostImport(this, nullptr);
		return nullptr;
	}

	TMap<FString, TConstArrayView64<uint8>> AdditionalBuffers;
	TArray64<uint8> WeightData;

	if (FileExtension.Compare(TEXT("xml"), ESearchCase::IgnoreCase) == 0)
	{
		const FString BinFilename(FPaths::ChangeExtension(Filename, "bin"));
		if (!FFileHelper::LoadFileToArray(WeightData, *BinFilename))
		{
			UE_LOG(LogNNERuntimeOpenVINOEditor, Error, TEXT("Failed to load additional binary xml data from file '%s'"), *BinFilename);
			GEditor->GetEditorSubsystem<UImportSubsystem>()->BroadcastAssetPostImport(this, nullptr);
			return nullptr;
		}
	}

	// FileData contains the XML model, WeightData contains the BIN weights.
	// Store everything into one contiguous blob for easy serialization in packaged builds.
	TArray64<uint8> SerializedFileData;
	FMemoryWriter64 MemoryWriter(SerializedFileData, true);
	int64 FileDataBytes = FileData.NumBytes();
	int64 WeightDataBytes = WeightData.NumBytes();
	MemoryWriter.Serialize(&FileDataBytes, sizeof(FileDataBytes));
	MemoryWriter.Serialize(&WeightDataBytes, sizeof(WeightDataBytes));
	MemoryWriter.Serialize(FileData.GetData(), FileDataBytes);
	MemoryWriter.Serialize(WeightData.GetData(), WeightDataBytes);

	UNNEModelData* ModelData = NewObject<UNNEModelData>(InParent, InClass, InName, Flags);
	check(ModelData)
	ModelData->Init(FileExtension, SerializedFileData);

	GEditor->GetEditorSubsystem<UImportSubsystem>()->BroadcastAssetPostImport(this, ModelData);

	return ModelData;
}

bool UNNERuntimeOpenVINOModelDataFactory::FactoryCanImport(const FString& Filename)
{
	const FString FileExtension(FPaths::GetExtension(Filename));
	if (!IsFileSupported(FileExtension))
	{
		return false;
	}

	if (FileExtension.Compare(TEXT("xml"), ESearchCase::IgnoreCase) == 0)
	{
		const FString BinFilename(FPaths::ChangeExtension(Filename, "bin"));
		if (!FPaths::FileExists(BinFilename))
		{
			UE_LOG(LogNNERuntimeOpenVINOEditor, Warning, TEXT("Skipping OpenVino IR import: Could not find binary file '%s' corresponding to '%s'"), *BinFilename, *Filename);
			return false;
		}
	}

	return true;
}
