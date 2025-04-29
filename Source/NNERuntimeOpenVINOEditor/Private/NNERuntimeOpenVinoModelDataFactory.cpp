// Copyright Epic Games, Inc. All Rights Reserved.

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
	TArray64<uint8> AdditionalData;

	if (FileExtension.Compare(TEXT("xml"), ESearchCase::IgnoreCase) == 0)
	{
		const FString BinFilename(FPaths::ChangeExtension(Filename, "bin"));
		if (!FFileHelper::LoadFileToArray(AdditionalData, *BinFilename))
		{
			UE_LOG(LogNNERuntimeOpenVINOEditor, Error, TEXT("Failed to load additional binary xml data from file '%s'"), *BinFilename);
			GEditor->GetEditorSubsystem<UImportSubsystem>()->BroadcastAssetPostImport(this, nullptr);
			return nullptr;
		}

		AdditionalBuffers.Emplace(TEXT(""), AdditionalData);
	}

	UNNEModelData* ModelData = NewObject<UNNEModelData>(InParent, InClass, InName, Flags);
	check(ModelData)
	ModelData->Init(FileExtension, FileData, AdditionalBuffers);

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
