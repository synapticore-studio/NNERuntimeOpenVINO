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

#pragma once

#include "CoreMinimal.h"

THIRD_PARTY_INCLUDES_START
#include "openvino/c/ov_core.h"
THIRD_PARTY_INCLUDES_END

#include "NNEModelData.h"
#include "NNEStatus.h"
#include "NNERuntimeRunSync.h"
#include "NNETypes.h"

bool IsFileSupported(const FString& FileType);

bool SupportsDevice(ov_core_t& OVInstance, const FString& BaseName);

ENNETensorDataType OpenVINOTypeToNNEType(ov_element_type_e ElementType);

void ReleasePorts(TArray<ov_output_const_port_t*>& Ports);

void ReleaseShapes(TArray<ov_shape_t>& Shapes);

void ReleasePartialShapes(TArray<ov_partial_shape_t>& Shapes);

void ReleaseTensors(TArray<ov_tensor_t*>& Tensors);

bool InitModelInstance(TSharedRef<UE::NNE::FSharedModelData> ModelData, TConstArrayView64<uint8> InAdditionalData, ov_model_t*& Model, ov_compiled_model_t*& CompiledModel, const FString& DeviceName);

bool InitModelTensorDescs(TArray<UE::NNE::FTensorDesc>& InDescs, TArray<UE::NNE::FTensorDesc>& OutDescs, ov_model_t*& Model);

UE::NNE::EResultStatus ModelInfer(TConstArrayView<UE::NNE::FTensorBindingCPU> InInputTensors, TConstArrayView<UE::NNE::FTensorBindingCPU> InOutputTensors, ov_model_t*& Model, ov_compiled_model_t*& CompiledModel);
