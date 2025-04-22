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

#include "Modules/ModuleInterface.h"

THIRD_PARTY_INCLUDES_START
#include "openvino/c/ov_core.h"
THIRD_PARTY_INCLUDES_END

DECLARE_LOG_CATEGORY_EXTERN(LogNNERuntimeOpenVINO, Log, All);

class UNNERuntimeOpenVINOCpu;
class UNNERuntimeOpenVINONpu;
class UNNERuntimeOpenVINOGpu;

class FNNERuntimeOpenVINO : public IModuleInterface
{
public:
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;

	ov_core_t& OpenVINOInstance() { return *OVCore; };

	static FName ModuleName();

private:
	TWeakObjectPtr<UNNERuntimeOpenVINOCpu> NNERuntimeOpenVINOCpu{ nullptr };
	TWeakObjectPtr<UNNERuntimeOpenVINONpu> NNERuntimeOpenVINONpu{ nullptr };
	TWeakObjectPtr<UNNERuntimeOpenVINOGpu> NNERuntimeOpenVINOGpu{ nullptr };

	// It's perfectly valid to create multiple ov_core instances in a process but it's recommended to only create one.
	ov_core_t* OVCore = nullptr;

	void LogDevices();
};
