# NNERuntimeOpenVINO
Intel's Unreal Engine OpenVINO plugin for NNE.

## Selective Device Build
By default the plugin includes support for all three NNE interfaces:
1. INNERuntimeCPU
2. INNERuntimeGPU
3. INNERuntimeNPU

The total distributable size in Release is around 100MB. If you do not intend to use a given interface it's possible to reduce this size by excluding the dynamic libraries for those interfaces. 

The device libraries can be found in the following location:

Binaries\openvino\\<build_type>

Device specific dynamic libraries are named as follows:
1. openvino_intel_cpu_plugin
2. openvino_intel_gpu_plugin
3. openvino_intel_npu_plugin

Simply delete the unwanted device libraries from the folder. The plugin build script will detect if a device library is present and selectively enable that interface. Please note that if you have already built your project and go back to remove one of these libraries, the build script cache will still attempt to search for it. In this case you either need to invalidate the plugin build script or rebuild your project.
