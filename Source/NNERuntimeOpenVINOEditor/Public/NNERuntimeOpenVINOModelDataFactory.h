// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "UObject/UObjectGlobals.h"
#include "Factories/Factory.h"

#include "NNERuntimeOpenVINOModelDataFactory.generated.h"

UCLASS()
class NNERUNTIMEOPENVINOEDITOR_API UNNERuntimeOpenVINOModelDataFactory : public UFactory
{
	GENERATED_BODY()

public:
	UNNERuntimeOpenVINOModelDataFactory(const FObjectInitializer& ObjectInitializer);

public:
	//~ Begin UFactory Interface
	virtual UObject* FactoryCreateFile(UClass* InClass, UObject* InParent, FName InName, EObjectFlags Flags, const FString& Filename, const TCHAR* Parms, FFeedbackContext* Warn, bool& bOutOperationCanceled) override;
	virtual bool FactoryCanImport(const FString& Filename) override;
	//~ End UFactory Interface

};