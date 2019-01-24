# GpuSpa
Welcome to the home repository of GpuSpa.

GpuSpa is a scalable GPU-based out-of-core graph system for interprocedural static analysis.

This Readme (under revision) provides a how-to-use guide for GpuSpa.
## Getting Started
### Required Libraries
Ensure that you have a recent version of the CUDA Toolkit and the Boost library installed in your system.
### Compiling GpuSpa
First, download the entire GpuSpa source code into your machine. Next, edit the src/makefile to set the paths to the CUDA include files and lib files in your machine. Finally, run the makefile in the src folder using make. GpuSpa should now be compiled and ready to run.
### Running GpuSpa
GpuSpa needs two input files: (1) a program graph on which GpuSpa can perform computations and (2) a grammar file which describes how the computations are to be performed. A sample input is inside the src/inputFiles folder.

After getting the graph and grammar file, run GpuSpa by entering the following command and monitor the progress of the computation on the screen:
```
../bin/comp <graph_file>  <grammar_file>
```
## Project Contributors
* [**Zhiqiang Zuo**](http://zuozhiqiang.bitbucket.io/) - *Assistant Professor, Nanjing University*
* **Shenming Lu** - *Master Student, Nanjing University*
