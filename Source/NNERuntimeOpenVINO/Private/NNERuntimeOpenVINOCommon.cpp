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

#include "NNERuntimeOpenVINOCommon.h"

#include "NNERuntimeOpenVINOModule.h"

bool IsFileSupported(const FString& FileType)
{
	/*
	* *.onnx -> NNERuntimeORT handles import
	* *.pb/*.pdmodel/*.tflite -> Must be converted externally using Python. There is no C/C++ support for this.
	* *.xml -> OpenVINO IR is the only one we need to handle.
	*/
	return (FileType.Compare(TEXT("onnx"), ESearchCase::IgnoreCase) == 0)
		|| (FileType.Compare(TEXT("xml"), ESearchCase::IgnoreCase) == 0);
}

bool SupportsDevice(ov_core_t& OVInstance, const FString& BaseName)
{
	ov_available_devices_t AvailableDevices{};
	if (ov_core_get_available_devices(&OVInstance, &AvailableDevices))
	{
		UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Failed to fetch OpenVINO devices."));
		return false;
	}

	for (size_t i = 0; i < AvailableDevices.size; ++i)
	{
		const FString DeviceName(AvailableDevices.devices[i]);
		if (DeviceName.Contains(BaseName))
		{
			ov_available_devices_free(&AvailableDevices);
			return true;
		}
	}

	ov_available_devices_free(&AvailableDevices);
	return false;
}

bool HasMultiGpu()
{
	// Load the model into OpenVINO
	FNNERuntimeOpenVINO* OVModule = FModuleManager::GetModulePtr<FNNERuntimeOpenVINO>(FNNERuntimeOpenVINO::ModuleName());
	if (!OVModule)
	{
		UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Couldn't find NNERuntimeOpenVINO module."));
		return false;
	}

	ov_core_t& OVCore = OVModule->OpenVINOInstance();

	ov_available_devices_t AvailableDevices{};
	if (ov_core_get_available_devices(&OVCore, &AvailableDevices))
	{
		UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Failed to fetch OpenVINO devices."));
		return false;
	}

	const FString BaseName(TEXT("GPU"));
	int32 NumGPUs = 0;
	for (size_t i = 0; i < AvailableDevices.size; ++i)
	{
		const FString DeviceName(AvailableDevices.devices[i]);
		if (DeviceName.Contains(BaseName))
		{
			NumGPUs++;
		}
	}

	ov_available_devices_free(&AvailableDevices);
	return NumGPUs > 1;
}

ENNETensorDataType OpenVINOTypeToNNEType(ov_element_type_e ElementType)
{
	switch (ElementType)
	{
	case ov_element_type_e::BOOLEAN:
		return ENNETensorDataType::Boolean;
	case ov_element_type_e::F16:
		return ENNETensorDataType::Half;
	case ov_element_type_e::F32:
		return ENNETensorDataType::Float;
	case ov_element_type_e::F64:
		return ENNETensorDataType::Double;
	case ov_element_type_e::I8:
		return ENNETensorDataType::Int8;
	case ov_element_type_e::I16:
		return ENNETensorDataType::Int16;
	case ov_element_type_e::I32:
		return ENNETensorDataType::Int32;
	case ov_element_type_e::I64:
		return ENNETensorDataType::Int64;
	case ov_element_type_e::U8:
		return ENNETensorDataType::UInt8;
	case ov_element_type_e::U16:
		return ENNETensorDataType::UInt16;
	case ov_element_type_e::U32:
		return ENNETensorDataType::UInt32;
	case ov_element_type_e::U64:
		return ENNETensorDataType::UInt64;
	case ov_element_type_e::BF16:
		return ENNETensorDataType::BFloat16;
	}
	return ENNETensorDataType::None;
}

void ReleasePorts(TArray<ov_output_const_port_t*>& Ports)
{
	for (ov_output_const_port_t*& Port : Ports)
	{
		if (Port)
		{
			ov_output_const_port_free(Port);
		}
	}
}

void ReleaseShapes(TArray<ov_shape_t>& Shapes)
{
	for (ov_shape_t& Shape : Shapes)
	{
		ov_shape_free(&Shape);
	}
}

void ReleasePartialShapes(TArray<ov_partial_shape_t>& Shapes)
{
	for (ov_partial_shape_t& Shape : Shapes)
	{
		ov_partial_shape_free(&Shape);
	}
}

void ReleaseTensors(TArray<ov_tensor_t*>& Tensors)
{
	for (ov_tensor_t*& Tensor : Tensors)
	{
		if (Tensor)
		{
			ov_tensor_free(Tensor);
		}
	}
}

bool InitModelInstance(TSharedRef<UE::NNE::FSharedModelData> ModelData, TConstArrayView64<uint8> InAdditionalData, ov_model_t*& Model, ov_compiled_model_t*& CompiledModel, const FString& DeviceName)
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
	if (!SupportsDevice(OVCore, DeviceName))
	{
		UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("[%s] device not found."), *DeviceName);
		return false;
	}

	ov_status_e LoadResult = ov_status_e::OK;
	if (InAdditionalData.IsEmpty())
	{
		LoadResult = ov_core_read_model_from_memory_buffer(&OVCore, (const char*)FileData.GetData(), FileData.Num(), NULL, &Model);
	}
	else 
	{
		// In the case of IR models, we must create a temporary Tensor as input.
		// The data for the tensor comes from the asset so there is no additional allocation inside of OpenVINO.
		ov_shape_t TempShape{};
		int64_t Dims[1] = { (int64_t)InAdditionalData.NumBytes() };
		if (ov_shape_create(1, Dims, &TempShape))
		{
			UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Failed to setup the IR model."));
			return false;
		}

		ov_tensor_t* TempTensor = NULL;
		if (ov_tensor_create_from_host_ptr(U8, TempShape, (void*)InAdditionalData.GetData(), &TempTensor))
		{
			ov_shape_free(&TempShape);
			UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Failed to setup the IR model."));
			return false;
		}

		LoadResult = ov_core_read_model_from_memory_buffer(&OVCore, (const char*)FileData.GetData(), FileData.Num(), TempTensor, &Model);
		
		ov_tensor_free(TempTensor);
		ov_shape_free(&TempShape);
	}

	if (LoadResult)
	{
		UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Failed to read the model."));
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

bool InitModelTensorDescs(TArray<UE::NNE::FTensorDesc>& InDescs, TArray<UE::NNE::FTensorDesc>& OutDescs, ov_model_t*& Model)
{
	size_t InputSize = 0;
	size_t OutputSize = 0;
	if (ov_model_inputs_size(Model, &InputSize) || ov_model_outputs_size(Model, &OutputSize))
	{
		UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Couldn't get input/Output size for model."));
		return false;
	}

	TArray<ov_output_const_port_t*> InputPorts;
	TArray<ov_partial_shape_t> InputPartialShapes;

	for (size_t i = 0; i < InputSize; ++i)
	{
		ov_output_const_port_t*& InputPort = InputPorts.AddZeroed_GetRef();
		if (ov_model_const_input_by_index(Model, i, &InputPort))
		{
			ReleasePorts(InputPorts);
			ReleasePartialShapes(InputPartialShapes);
			InDescs.Empty();
			UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Failed to get input config."));
			return false;
		}

		ov_element_type_e InputType{};
		if (ov_port_get_element_type(InputPort, &InputType))
		{
			ReleasePorts(InputPorts);
			ReleasePartialShapes(InputPartialShapes);
			InDescs.Empty();
			UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Failed to get input type."));
			return false;
		}

		ov_partial_shape_t& InputShape = InputPartialShapes.AddZeroed_GetRef();
		if (ov_port_get_partial_shape(InputPort, &InputShape))
		{
			ReleasePorts(InputPorts);
			ReleasePartialShapes(InputPartialShapes);
			InDescs.Empty();
			UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Failed to get input partial shape."));
			return false;
		}

		ENNETensorDataType DataType = OpenVINOTypeToNNEType(InputType);

		TArray<int32> SymbolicShape;
		for (int32 j = 0; j < InputShape.rank.min; j++)
		{
			if (ov_dimension_is_dynamic(InputShape.dims[j]) || InputShape.dims[j].min != InputShape.dims[j].max)
			{
				SymbolicShape.Add(-1);
			}
			else
			{
				SymbolicShape.Add(InputShape.dims[j].min);
			}
		}

		UE::NNE::FTensorDesc TensorDesc = UE::NNE::FTensorDesc::Make("", UE::NNE::FSymbolicTensorShape::Make(SymbolicShape), DataType);
		InDescs.Add(TensorDesc);
	}

	ReleasePorts(InputPorts);
	ReleasePartialShapes(InputPartialShapes);

	TArray<ov_output_const_port_t*> OutputPorts;
	TArray<ov_partial_shape_t> OutputPartialShapes;

	for (size_t i = 0; i < OutputSize; ++i)
	{
		ov_output_const_port_t*& OutputPort = OutputPorts.AddZeroed_GetRef();
		if (ov_model_const_output_by_index(Model, i, &OutputPort))
		{
			ReleasePorts(OutputPorts);
			ReleasePartialShapes(OutputPartialShapes);
			OutDescs.Empty();
			UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Failed to get output config."));
			return false;
		}

		ov_element_type_e OutputType{};
		if (ov_port_get_element_type(OutputPort, &OutputType))
		{
			ReleasePorts(OutputPorts);
			ReleasePartialShapes(OutputPartialShapes);
			OutDescs.Empty();
			UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Failed to get output type."));
			return false;
		}

		ov_partial_shape_t& OutputShape = OutputPartialShapes.AddZeroed_GetRef();
		if (ov_port_get_partial_shape(OutputPort, &OutputShape))
		{
			ReleasePorts(OutputPorts);
			ReleasePartialShapes(OutputPartialShapes);
			OutDescs.Empty();
			UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Failed to get output partial shape."));
			return false;
		}

		ENNETensorDataType DataType = OpenVINOTypeToNNEType(OutputType);

		TArray<int32> SymbolicShape;
		for (int32 j = 0; j < OutputShape.rank.min; j++)
		{
			if (ov_dimension_is_dynamic(OutputShape.dims[j]) || OutputShape.dims[j].min != OutputShape.dims[j].max)
			{
				SymbolicShape.Add(-1);
			}
			else
			{
				SymbolicShape.Add(OutputShape.dims[j].min);
			}
		}

		UE::NNE::FTensorDesc TensorDesc = UE::NNE::FTensorDesc::Make("", UE::NNE::FSymbolicTensorShape::Make(SymbolicShape), DataType);
		OutDescs.Add(TensorDesc);
	}

	ReleasePorts(OutputPorts);
	ReleasePartialShapes(OutputPartialShapes);

	return true;
}

UE::NNE::EResultStatus ModelInfer(TConstArrayView<UE::NNE::FTensorBindingCPU> InInputTensors, TConstArrayView<UE::NNE::FTensorBindingCPU> InOutputTensors, ov_model_t*& Model, ov_compiled_model_t*& CompiledModel)
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
