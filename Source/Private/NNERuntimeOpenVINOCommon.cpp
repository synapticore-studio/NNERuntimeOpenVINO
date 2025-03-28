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

