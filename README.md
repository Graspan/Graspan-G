# Graspan-G
Welcome to the home repository of Graspan-G.

Graspan-G is a scalable GPU-based out-of-core graph system for interprocedural static analysis.

This Readme (under revision) provides a how-to-use guide for Graspan-G.
## Getting Started
### Required Libraries
Ensure that you have a recent version(10.2 in our project) of the CUDA Toolkit and the Boost library installed in your system.
### Compiling Graspan-G
First, download the entire Graspan-G source code into your machine. Next, edit the src/makefile to set the paths to the CUDA include files and lib files in your machine. Finally, run the makefile in the src folder using make. Graspan-G should now be compiled and ready to run.
### Running Graspan-G
Graspan-G needs two input files: 
1. A program graph on which Graspan-G can perform computations;
2. A grammar file which describes how the computations are to be performed. 

A sample input is inside the src/inputFiles folder.

After getting the graph and grammar file, run Graspan-G by entering the following command and monitor the progress of the computation on the screen:
```
../bin/comp <graph_file>  <grammar_file>  <filter_flag> 
```

Note that the filter flag is a switch to optimize the performance of dataflow analysis rules only. 0 means off optimization, 1 means on optimization. In other analysis, the flag should be set to 0.

## Project Contributors
* [**Zhiqiang Zuo**](https://z-zhiqiang.github.io/) - *Assistant Professor, Nanjing University*
* **Yiyu Zhang** - *PhD Student, Nanjing University*
* **Shenming Lu** - *Master Student, Nanjing University*
