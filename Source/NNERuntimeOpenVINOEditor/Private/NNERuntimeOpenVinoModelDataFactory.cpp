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
	return (FileType.Compare(TEXT("onnx"), ESearchCase::IgnoreCase) == 0)
		|| (FileType.Compare(TEXT("pb"), ESearchCase::IgnoreCase) == 0)
		|| (FileType.Compare(TEXT("pdmodel"), ESearchCase::IgnoreCase) == 0)
		|| (FileType.Compare(TEXT("tflite"), ESearchCase::IgnoreCase) == 0)
		|| (FileType.Compare(TEXT("xml"), ESearchCase::IgnoreCase) == 0);
}

UNNERuntimeOpenVINOModelDataFactory::UNNERuntimeOpenVINOModelDataFactory(const FObjectInitializer& ObjectInitializer) : UFactory(ObjectInitializer)
{
	bCreateNew = false;
	bEditorImport = true;
	SupportedClass = UNNEModelData::StaticClass();
	ImportPriority = DefaultImportPriority;
	Formats.Add("xml;OpenVINO IR Format");
	Formats.Add("pb;Tensorflow Format");
	Formats.Add("tflite;Tensorflow Lite Format");
	Formats.Add("pdmodel;PDPD Format");
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

	if (FileExtension.Compare(TEXT("xml"), ESearchCase::IgnoreCase) == 0)
	{
		TArray64<uint8> AdditionalData;

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
