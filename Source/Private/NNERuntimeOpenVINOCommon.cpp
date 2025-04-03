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
	return (FileType.Compare("onnx", ESearchCase::IgnoreCase) == 0)
		|| (FileType.Compare("pdmodel", ESearchCase::IgnoreCase) == 0)
		|| (FileType.Compare("tflite", ESearchCase::IgnoreCase) == 0);
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

bool InitModelInstance(TSharedRef<UE::NNE::FSharedModelData> ModelData, ov_model_t*& Model, ov_compiled_model_t*& CompiledModel, const FString& DeviceName)
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
	if (ov_core_read_model_from_memory_buffer(&OVCore, (const char*)FileData.GetData(), FileData.Num(), NULL, &Model))
	{
		UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Failed to read the model."));
		return false;
	}

	if (!SupportsDevice(OVCore, DeviceName))
	{
		ov_model_free(Model);
		Model = nullptr;
		UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("[%s] device not found."), *DeviceName);
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

	FNNERuntimeOpenVINO* OVModule = FModuleManager::GetModulePtr<FNNERuntimeOpenVINO>(FNNERuntimeOpenVINO::ModuleName());
	if (!OVModule)
	{
		UE_LOG(LogNNERuntimeOpenVINO, Error, TEXT("Couldn't find NNERuntimeOpenVINO module."));
		return UE::NNE::EResultStatus::Fail;
	}

	ov_core_t& OVInstance = OVModule->OpenVINOInstance();

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

