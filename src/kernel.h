#ifndef KERNEL_H
#define KERNEL_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <limits.h>
#include <string.h>
#include <ctime>

typedef unsigned int uint;
typedef unsigned char uchar;

extern int selectedDevice;
extern int numLabels;
extern uint numVars;

#define DEVMEMUSED_MB 10000

#define COMPRESSED 1

#define NUMTHREADS 8

#define ELEMENT_WIDTH 32
#define ELEMENT_CARDINALITY ((ELEMENT_WIDTH - 2) * 32)
#define BASE (ELEMENT_WIDTH - 2)
#define NEXT (ELEMENT_WIDTH - 1)

#define NIL UINT_MAX

#define WARP_SIZE 32
#define THREADS_PER_BLOCK 768

#define BASE_OF(x) (x / ELEMENT_CARDINALITY)
#define UNIT_OF(x) (div32(x % ELEMENT_CARDINALITY))
#define BIT_OF(x) (mod32(x))

#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost
#define D2D cudaMemcpyDeviceToDevice

#define B2MB(x) (x / (1024 * 1024))
#define MB2B(x) (x * 1024 * 1024)

#define cudaSafeCall(err) { \
if (err != cudaSuccess) { \
printf("Runtime API error: %s.\n", cudaGetErrorString(err)); \
exit(-1); \
} \
}

struct Partition {
	// interval
	uint firstVar, lastVar;
	// avoid duplicates
	uint oldSize, deltaSize, tmpSize;
	// edge array or sparse bit vector
	bool flag;
	Partition(uint a, uint b, uint c, uint d, uint e, bool f) : firstVar(a), lastVar(b), oldSize(c), deltaSize(d), tmpSize(e), flag(f) {}
};

struct ComputeRegion {
	uint firstVar;
	uint lastVar;
	int start;
	uint offset;
	bool flag;
	ComputeRegion(uint a, uint b, int s, uint o, bool f) : firstVar(a), lastVar(b), start(s), offset(o), flag(f) {}
};

__device__ __host__ inline uint mul32(uint num)
{
	return num << 5;
}

__device__ __host__ inline uint div32(uint num)
{
	return num >> 5;
}

__device__ __host__ inline uint mod32(uint num)
{
	return num & 31;
}

// e.g. for powerOfTwo = 32: 0 => 0, 4 => 0, 32 => 32, 33 => 32
// second parameter has to be a power of two
__device__ __host__ inline uint roundToPrevMultipleOf(uint num, uint powerOfTwo = 32)
{
	if ((num & (powerOfTwo - 1)) == 0) return num;
	return (num / powerOfTwo + 1) * powerOfTwo - 32;
}

// e.g. for powerOfTwo = 32: 4 => 32, 32 => 32, 33 => 64
// second parameter has to be a power of two
__device__ __host__ inline uint roundToNextMultipleOf(uint num, uint powerOfTwo = 32)
{
	if ((num & (powerOfTwo - 1)) == 0) return num;
	return (num / powerOfTwo + 1) * powerOfTwo;
}

__device__ __host__ inline uint getElapsedTime(clock_t startTime)
{
	return (clock() - startTime) / (CLOCKS_PER_SEC * 60);
}

void transferRules(int numRules, int *rules);

void allocateElementPool(uint &heapSize, uint* &elementPool);

void transferElements(Partition p, uint *elements, int start, uint heapSize, uint *elementPool);

void spagpu_s(Partition &p, int start, uint heapSize, uint *elementPool, bool &r);

void spagpu_b(Partition &p1, Partition &p2, bool &r1, bool &r2, uint heapSize, uint *elementPool);

void spagpu(Partition &p1, Partition &p2, bool &r1, bool &r2, uint heapSize);

void getDegree(Partition p, int start, uint *degree);

void mergeAndDiff(Partition &p, uint heapSize, uint *elementPool);

#endif
