
# CudaEngine

C++ Terrain generation and rendering engine with optional hardware acceleration.

## Deprecated

This repository has been discontinued and split into:
[HardwareNoise](https://github.com/ChrisJMurdoch/HardwareNoise), 
[GLRender](https://github.com/ChrisJMurdoch/GLRender), and 
[CppUtility](https://github.com/ChrisJMurdoch/CppUtility).

## Requirements

The **Microsoft Visual C++ Redistributable** is required to run the program.
It can be downloaded at: https://support.microsoft.com/en-gb/help/2977003

A **GLFW** dll is included but can be built from: https://www.glfw.org

**NVIDIA Tesla** and up is required to run with hardware acceleration.

## Running

To run the engine with hardware acceleration, no command line arguments are required.

To use the acceleration features, use "cuda" as the first command line argument.
