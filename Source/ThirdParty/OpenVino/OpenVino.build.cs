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

using System.IO;
using System.Collections.Generic;
using Microsoft.Extensions.Logging;
using UnrealBuildTool;

public class OpenVino : ModuleRules
{
	public OpenVino(ReadOnlyTargetRules Target) : base(Target)
	{
		Type = ModuleType.External;

		// Include paths for OpenVINO.
		string OpenVinoIncludePath = Path.Combine(ModuleDirectory, "Internal");
		PublicIncludePaths.Add(OpenVinoIncludePath);

		// OpenVINO/TBB Dlls
		List<string> OpenVINOLibs = new List<string>();
		string ConfigName = Target.Configuration == UnrealTargetConfiguration.Debug ? "Debug" : "Release";
		string OpenVINOSubDirPath = Path.Combine("Binaries", "openvino", Target.Platform.ToString(), ConfigName);
		string OpenVINOLibPath = Path.Combine(PluginDirectory, OpenVINOSubDirPath);
		string TbbSubDirPath = Path.Combine(OpenVINOSubDirPath, "tbb");
		OpenVINOSubDirPath = OpenVINOSubDirPath.Replace(@"\", @"/");
		TbbSubDirPath = TbbSubDirPath.Replace(@"\", @"/");
		PublicDefinitions.Add($"OPENVINO_PATH=\"{OpenVINOSubDirPath}\"");
		PublicDefinitions.Add($"OPENVINO_TBB_PATH=\"{TbbSubDirPath}\"");

		// Link to *.lib or *.so to resolve external C symbols.
		string TbbDllPath = Path.Combine(OpenVINOLibPath, "tbb");
		if (Target.Configuration == UnrealTargetConfiguration.Debug)
		{
			if (Target.Platform == UnrealTargetPlatform.Win64)
			{
				OpenVINOLibs.Add(Path.Combine(OpenVINOLibPath, "openvino_cd.lib"));

				List<IEnumerable<string>> PDBCollection = new List<IEnumerable<string>> {
					Directory.EnumerateFiles(OpenVINOLibPath, "*.pdb"),
					Directory.EnumerateFiles(TbbDllPath, "*.pdb")
				};

				foreach (IEnumerable<string> Collection in PDBCollection)
				{
					foreach (string PDB in Collection)
					{
						RuntimeDependencies.Add(PDB);
					}
				}
			}
			else if (Target.Platform == UnrealTargetPlatform.Linux)
			{
				// Linux Debug libraries follow the same naming convention as the Release version.
				OpenVINOLibs.Add(Path.Combine(OpenVINOLibPath, "openvino_c"));
			}
		}
		else
		{
			if (Target.Platform == UnrealTargetPlatform.Win64)
			{
				OpenVINOLibs.Add(Path.Combine(OpenVINOLibPath, "openvino_c.lib"));
			}
			else if (Target.Platform == UnrealTargetPlatform.Linux)
			{
				OpenVINOLibs.Add(Path.Combine(OpenVINOLibPath, "openvino_c"));
			}
		}

		// Search OpenVINO Dlls and TBB Dlls.
		// Only need the base files on Linux, we don't copy versioned to the runtime directory.
		string PlatformDLLExtension = (Target.Platform == UnrealTargetPlatform.Win64) ? "*.dll" : "*.so";

		List<IEnumerable<string>> DllCollection = new List<IEnumerable<string>> {
			Directory.EnumerateFiles(OpenVINOLibPath, PlatformDLLExtension),
			Directory.EnumerateFiles(TbbDllPath, PlatformDLLExtension)
		};

		PublicAdditionalLibraries.AddRange(OpenVINOLibs);

		// Check for device plugins. If a device plugin DLL is not found, do not initialize that interface.
		bool bHasCPUPlugin = false;
		bool bHasGPUPlugin = false;
		bool bHasNPUPlugin = false;

		// Stage all the Dlls to the same ouput destination.
		foreach (IEnumerable<string> Collection in DllCollection)
		{
			foreach (string Dll in Collection)
			{
				string DLLFileName = Path.GetFileName(Dll);
				RuntimeDependencies.Add(Dll);
				PublicDelayLoadDLLs.Add(DLLFileName);

				if (DLLFileName.Contains("cpu_plugin"))
				{
					bHasCPUPlugin = true;
				}
				else if (DLLFileName.Contains("gpu_plugin"))
				{
					bHasGPUPlugin = true;
				}
				else if (DLLFileName.Contains("npu_plugin"))
				{
					bHasNPUPlugin = true;
				}
				else if(DLLFileName.Contains("openvino_c"))
				{
					// Special case for the base DLL we need to manually load.
					PublicDefinitions.Add($"OPENVINO_DLL_NAME=\"{DLLFileName}\"");
				}
			}
		}

		if (bHasCPUPlugin)
		{
			PublicDefinitions.Add("OPENVINO_CPU_PLUGIN");
		}

		if (bHasGPUPlugin)
		{
			PublicDefinitions.Add("OPENVINO_GPU_PLUGIN");
		}

		if (bHasNPUPlugin)
		{
			PublicDefinitions.Add("OPENVINO_NPU_PLUGIN");
		}

		if (!bHasCPUPlugin && !bHasGPUPlugin && !bHasNPUPlugin)
		{
			Logger.LogWarning("No Device plugins found. No interfaces will be available.");
		}
	}
}