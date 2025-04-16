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

using System.Collections.Generic;
using System.IO;
using Microsoft.Extensions.Logging;
using UnrealBuildTool;

public class NNERuntimeOpenVINO : ModuleRules
{
	public NNERuntimeOpenVINO(ReadOnlyTargetRules Target) : base(Target)
	{
		int EngineMajorVersion = ReadOnlyBuildVersion.Current.MajorVersion;
		int EngineMinorVersion = ReadOnlyBuildVersion.Current.MinorVersion;

		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PrivateDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				"CoreUObject",
				"Engine",
				"NNE"
			}
		);

		// Include paths for OpenVINO.
		string OpenVINOIncludePath = Path.Combine(ModuleDirectory, "..", "openvino");
		PrivateIncludePaths.Add(OpenVINOIncludePath);

		// OpenVINO/TBB Dlls
		string OpenVINOLibPath = Path.Combine(PluginDirectory, "Binaries", "openvino");
		List<string> OpenVINOLibs = new List<string>();

		if (Target.Configuration == UnrealTargetConfiguration.Debug)
		{
			OpenVINOLibPath = Path.Combine(OpenVINOLibPath, "Debug");
			OpenVINOLibs.Add(Path.Combine(OpenVINOLibPath, "openvino_cd.lib"));
		}
		else
		{
			OpenVINOLibPath = Path.Combine(OpenVINOLibPath, "Release");
			OpenVINOLibs.Add(Path.Combine(OpenVINOLibPath, "openvino_c.lib"));
		}

		// Search OpenVINO Dlls and TBB Dlls.
		List<IEnumerable<string>> DllCollection = new List<IEnumerable<string>>();
		DllCollection.Add(Directory.EnumerateFiles(OpenVINOLibPath, "*.dll"));

		string TbbDllPath = Path.Combine(OpenVINOLibPath, "tbb");
		DllCollection.Add(Directory.EnumerateFiles(TbbDllPath, "*.dll"));

		PublicAdditionalLibraries.AddRange(OpenVINOLibs);

		// Check for device plugins. If a device plugin DLL is not found, do not initialize that interface.
		bool bHasCPUPlugin = false;
		bool bHasGPUPlugin = false;
		bool bHasNPUPlugin = false;

		// Stage all the Dlls to the same ouput destination.
		string OutputBasePath = Path.Combine("$(TargetOutputDir)", "OpenVINO");
		foreach(IEnumerable<string> Collection in DllCollection)
		{
			foreach(string Dll in Collection)
			{
				string DLLFileName = Path.GetFileName(Dll);
				RuntimeDependencies.Add(Path.Combine(OutputBasePath, DLLFileName), Dll);

				if(DLLFileName.Contains("cpu_plugin"))
				{
					bHasCPUPlugin = true;
				}
				else if(DLLFileName.Contains("gpu_plugin"))
				{
					bHasGPUPlugin = true;
				}
				else if(DLLFileName.Contains("npu_plugin"))
				{
					bHasNPUPlugin = true;
				}
			}
		}

		if(bHasCPUPlugin)
		{
			PrivateDefinitions.Add("OPENVINO_CPU_PLUGIN");
		}

		if (bHasGPUPlugin)
		{
			PrivateDefinitions.Add("OPENVINO_GPU_PLUGIN");
		}

		if (bHasNPUPlugin)
		{
			PrivateDefinitions.Add("OPENVINO_NPU_PLUGIN");
		}

		if (!bHasCPUPlugin && !bHasGPUPlugin && !bHasNPUPlugin)
		{
			Logger.LogWarning("No Device plugins found. None of the interfaces will be available.");
		}
	}
}
