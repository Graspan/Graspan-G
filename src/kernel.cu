#include "kernel.h"

int selectedDevice;
const int N = 20;
int numLabels;
uint numVars;

__device__ __constant__ int dev_numRules;
__device__ __constant__ int dev_rules[N * 3];
__device__ __constant__ int dev_numLabels;
__device__ __constant__ uint dev_heapSize;
__device__ __constant__ uint *dev_elementPool;
// index of the next free element in the corresponding free list
// in words
__device__ uint freeList1 = 0;
__device__ uint freeList2 = 0;
__device__ uint worklistIndex = 0;
__device__ uint counter = 0;
__device__ bool dev_r = false;
__device__ bool dev_changed = false;

void transferRules(int numRules, int *rules)
{
	if (numRules > N) {
		std::cerr << "N needs to be reset." << std::endl;
		exit(-1);
	}
	cudaSafeCall(cudaMemcpyToSymbol(dev_numRules, &numRules, sizeof(int)));
	cudaSafeCall(cudaMemcpyToSymbol(dev_rules, rules, numRules * sizeof(int) * 3));
	cudaSafeCall(cudaMemcpyToSymbol(dev_numLabels, &numLabels, sizeof(int)));
}

void allocateElementPool(uint &heapSize, uint* &elementPool)
{
	heapSize = (DEVMEMUSED_MB / 8 * 7) * 1024 * 256;
	std::cout << "HEAP SIZE: " << heapSize << " (in 32-bit words)" << std::endl;
	cudaSafeCall(cudaMalloc((void **)&elementPool, heapSize * sizeof(uint)));
	cudaSafeCall(cudaMemcpyToSymbol(dev_heapSize, &heapSize, sizeof(uint)));
	cudaSafeCall(cudaMemcpyToSymbol(dev_elementPool, &elementPool, sizeof(uint*)));
}

void transferElements(Partition p, uint *elements, int start, uint heapSize, uint *elementPool)
{
	uint poolSize = p.oldSize + p.deltaSize + p.tmpSize;
	if (start) {
		cudaSafeCall(cudaMemcpyToSymbol(freeList2, &poolSize, sizeof(uint)));
	} else {
		cudaSafeCall(cudaMemcpyToSymbol(freeList1, &poolSize, sizeof(uint)));
	}
	cudaSafeCall(cudaMemcpy(elementPool + (heapSize / 2) * start, elements, poolSize * sizeof(uint), H2D));
	delete[] elements;
}

void initialize(uint headSize, int start, uint offset, uint heapSize, uint *elementPool)
{
	uint poolSize = offset + headSize;
	if (start) {
		cudaSafeCall(cudaMemcpyToSymbol(freeList2, &poolSize, sizeof(uint)));
	} else {
		cudaSafeCall(cudaMemcpyToSymbol(freeList1, &poolSize, sizeof(uint)));
	}
	cudaSafeCall(cudaMemset(elementPool + (heapSize / 2) * start + offset, -1, headSize * sizeof(uint)));
}

void needRepartition(Partition &p, uint heapSize, bool &r)
{
	bool s = false;
	cudaSafeCall(cudaMemcpyFromSymbol(&s, dev_r, sizeof(bool)));
	if (s) {
		p.tmpSize = heapSize / 2 - p.deltaSize - p.oldSize;
		s = false;
		cudaSafeCall(cudaMemcpyToSymbol(dev_r, &s, sizeof(bool)));
		r = true;
		std::cout << "Need Repartition." << std::endl;
	}
}

__host__ inline uint getBlocks()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, selectedDevice);
	return deviceProp.multiProcessorCount * 4;
}

__device__ inline uint getThreadIdInBlock()
{
	return threadIdx.x + threadIdx.y * blockDim.x;
}

__device__ inline uint isFirstThreadOfBlock()
{
	return !getThreadIdInBlock();
}

__device__ inline uint isFirstThreadOfVWarp()
{
	return !threadIdx.x;
}

__device__ inline void graphSet(const uint pos, const uint val)
{
	dev_elementPool[pos] = val;
}

__device__ inline uint graphGet(const uint pos)
{
	return dev_elementPool[pos];
}

__device__ inline uint getAndIncrement(const uint delta)
{
	__shared__ volatile uint temp[THREADS_PER_BLOCK / ELEMENT_WIDTH];
	if (isFirstThreadOfVWarp())
		temp[threadIdx.y] = atomicAdd(&worklistIndex, delta);
	return temp[threadIdx.y];
}

__device__ inline void resetWorklistIndex()
{
	__syncthreads();
	if (isFirstThreadOfBlock() && atomicInc(&counter, gridDim.x - 1) == (gridDim.x - 1))
		worklistIndex = 0;
}

__device__ uint getValAtThread(const uint myVal, const uint i)
{
	__shared__ volatile uint temp[THREADS_PER_BLOCK / ELEMENT_WIDTH];
	if (threadIdx.x == i)
		temp[threadIdx.y] = myVal;
	return temp[threadIdx.y];
}

__device__ inline uint mallocIn(int start, uint size = ELEMENT_WIDTH)
{
	__shared__ volatile uint temp[THREADS_PER_BLOCK / ELEMENT_WIDTH];
	if (isFirstThreadOfVWarp()) {
		if (start) {
			temp[threadIdx.y] = atomicAdd(&freeList2, size);
		} else {
			temp[threadIdx.y] = atomicAdd(&freeList1, size);
		}
	}
	if (temp[threadIdx.y] + size > dev_heapSize / 2) {
		dev_r = true;
		return -1;
	} else {
		return temp[threadIdx.y];
	}
}

__device__ inline uint getIndex(uint headIndex, int start)
{
	uint index = graphGet(headIndex);
	if (index == NIL) {
		uint newIndex = mallocIn(start);
		if (newIndex != -1) {
			graphSet((dev_heapSize / 2) * start + newIndex + threadIdx.x, NIL);
			graphSet(headIndex, newIndex);
		}
		return newIndex;
	}
	return index;
}

__device__ uint addElement(uint index, uint fromBase, uint fromBits, int start)
{
	uint startIndex = (dev_heapSize / 2) * start;
	for (;;) {
		uint toBits = graphGet(index + threadIdx.x);
		uint toBase = getValAtThread(toBits, BASE);
		if (toBase == NIL) {
			// can only happen if the list is empty
			graphSet(index + threadIdx.x, fromBits);
			return index;
		}
		if (toBase == fromBase) {
			uint orBits = toBits | fromBits;
			if (orBits != toBits && threadIdx.x < NEXT)
				graphSet(index + threadIdx.x, orBits);
			return index;
		}
		if (toBase < fromBase) {
			uint toNext = getValAtThread(toBits, NEXT);
			if (toNext == NIL) {
				// appending
				uint newIndex = mallocIn(start);
				if (newIndex == -1) return -1;
				graphSet(newIndex + startIndex + threadIdx.x, fromBits);
				graphSet(index + NEXT, newIndex);
				return newIndex + startIndex;
			}
			index = toNext + startIndex;
		} else {
			uint newIndex = mallocIn(start);
			if (newIndex == -1) return -1;
			graphSet(newIndex + startIndex + threadIdx.x, toBits);
			uint val = threadIdx.x == NEXT ? newIndex : fromBits;
			graphSet(index + threadIdx.x, val);
			return index;
		}
	}
}

__device__ uint insert(uint index, uint var, int start)
{
	uint base = BASE_OF(var);
	uint unit = UNIT_OF(var);
	uint bit = BIT_OF(var);
	uint myBits = 0;
	if (threadIdx.x == unit) myBits = 1 << bit;
	if (threadIdx.x == BASE) myBits = base;
	if (threadIdx.x == NEXT) myBits = NIL;
	return addElement(index, base, myBits, start);
}

__device__ uint clone(uint nextIndex, int toStart, uint fromBits, uint fromNext, uint fromStartIndex)
{
	uint toStartIndex = (dev_heapSize / 2) * toStart;
	for (;;) {
		uint newIndex = mallocIn(toStart);
		if (newIndex == -1) return -1;
		dev_changed = true;
		uint val = threadIdx.x == NEXT ? NIL : fromBits;
		graphSet(newIndex + toStartIndex + threadIdx.x, val);
		graphSet(nextIndex, newIndex);
		if (fromNext == NIL) break;
		fromBits = graphGet(fromNext + fromStartIndex + threadIdx.x);
		fromNext = getValAtThread(fromBits, NEXT);
		nextIndex = newIndex + toStartIndex + NEXT;
	}
	return 0;
}

__device__ uint union2(uint to, uint toRel, ComputeRegion tmp, uint fromIndex, int fromStart)
{
	uint fromStartIndex = (dev_heapSize / 2) * fromStart;
	uint toStartIndex = (dev_heapSize / 2) * tmp.start;
	uint fromBits = graphGet(fromIndex + threadIdx.x);
	uint fromBase = getValAtThread(fromBits, BASE);
	uint fromNext = getValAtThread(fromBits, NEXT);
	uint toHeadIndex = toStartIndex + tmp.offset + roundToNextMultipleOf(tmp.lastVar - tmp.firstVar + 1) * (toRel - 1) + to;
	uint toIndex = graphGet(toHeadIndex);
	if (toIndex == NIL) {
		uint s = clone(toHeadIndex, tmp.start, fromBits, fromNext, fromStartIndex);
		if (s == -1) return -1;
		return 0;
	}
	toIndex += toStartIndex;
	uint toBits = graphGet(toIndex + threadIdx.x);
	uint toBase = getValAtThread(toBits, BASE);
	uint toNext = getValAtThread(toBits, NEXT);
	for (;;) {
		if (toBase > fromBase) {
			uint newIndex = mallocIn(tmp.start);
			if (newIndex == -1) return -1;
			dev_changed = true;
			graphSet(newIndex + toStartIndex + threadIdx.x, toBits);
			uint val = threadIdx.x == NEXT ? newIndex : fromBits;
			graphSet(toIndex + threadIdx.x, val);
			if (fromNext == NIL) return 0;
			toIndex = newIndex + toStartIndex;
			fromBits = graphGet(fromNext + fromStartIndex + threadIdx.x);
			fromBase = getValAtThread(fromBits, BASE);
			fromNext = getValAtThread(fromBits, NEXT);
		} else if (toBase == fromBase) {
			uint orBits = fromBits | toBits;
			uint newBits = threadIdx.x == NEXT ? toNext : orBits;
			if (newBits != toBits) dev_changed = true;
			graphSet(toIndex + threadIdx.x, newBits);
			if (fromNext == NIL) return 0;
			fromBits = graphGet(fromNext + fromStartIndex + threadIdx.x);
			fromBase = getValAtThread(fromBits, BASE);
			fromNext = getValAtThread(fromBits, NEXT);
			if (toNext == NIL) {
				uint s = clone(toIndex + NEXT, tmp.start, fromBits, fromNext, fromStartIndex);
				if (s == -1) return -1;
				return 0;
			}
			toIndex = toNext + toStartIndex;
			toBits = graphGet(toIndex + threadIdx.x);
			toBase = getValAtThread(toBits, BASE);
			toNext = getValAtThread(toBits, NEXT);
		} else {
			if (toNext == NIL) {
				uint s = clone(toIndex + NEXT, tmp.start, fromBits, fromNext, fromStartIndex);
				if (s == -1) return -1;
				return 0;
			}
			toIndex = toNext + toStartIndex;
			toBits = graphGet(toIndex + threadIdx.x);
			toBase = getValAtThread(toBits, BASE);
			toNext = getValAtThread(toBits, NEXT);
		}
	}
}

__device__ uint unionAll(uint toRel, uint fromRel, uint to, uint numFroms, uint *p, ComputeRegion dst1, ComputeRegion dst2, ComputeRegion tmp)
{
	uint startIndex_dst1 = (dev_heapSize / 2) * dst1.start;
	uint virtualNumPartialVars_dst1 = roundToNextMultipleOf(dst1.lastVar - dst1.firstVar + 1);
	uint startIndex_dst2 = (dev_heapSize / 2) * dst2.start;
	uint virtualNumPartialVars_dst2 = roundToNextMultipleOf(dst2.lastVar - dst2.firstVar + 1);
	for (uint i = 0; i < numFroms; i++) {
		if (p[i] >= dst1.firstVar && p[i] <= dst1.lastVar) {
			uint headIndex1 = startIndex_dst1 + dst1.offset + virtualNumPartialVars_dst1 * (fromRel - 1) + p[i] - dst1.firstVar;
			uint fromIndex1 = graphGet(headIndex1);
			if (fromIndex1 != NIL) {
				uint s = union2(to, toRel, tmp, fromIndex1 + startIndex_dst1, dst1.start);
				if (s == -1) return -1;
			}
		}
		if (dst2.flag) {
			if (p[i] >= dst2.firstVar && p[i] <= dst2.lastVar) {
				uint headIndex2 = startIndex_dst2 + dst2.offset + virtualNumPartialVars_dst2 * (fromRel - 1) + p[i] - dst2.firstVar;
				uint fromIndex2 = graphGet(headIndex2);
				if (fromIndex2 != NIL) {
					uint s = union2(to, toRel, tmp, fromIndex2 + startIndex_dst2, dst2.start);
					if (s == -1) return -1;
				}
			}
		}
	}
	return 0;
}

__device__  uint decode(uint toRel, uint fromRel, uint myBits, uint base, uint i, uint *p, ComputeRegion dst1, ComputeRegion dst2, ComputeRegion tmp)
{
	for (int j = 0; j < BASE; j++) {
		uint bits = getValAtThread(myBits, j);
		if (bits) {
			uint numOnes = __popc(bits);
			for (int k = 0; k < 32 / blockDim.x; k++) {
				uint threadId = threadIdx.x + blockDim.x * k;
				uint threadMask = 1 << threadId;
				uint myMask = threadMask - 1;
				uint var = base * ELEMENT_CARDINALITY + mul32(j) + threadId;
				uint bitActive = bits & threadMask;
				uint pos = __popc(bits & myMask);
				if (bitActive) p[pos] = var;
			}
			uint s = unionAll(toRel, fromRel, i, numOnes, p, dst1, dst2, tmp);
			if (s == -1) return -1;
		}
	}
	return 0;
}

__device__ uint apply(uint firstRel, uint secondRel, uint thirdRel, uint i, uint *p, ComputeRegion src, ComputeRegion dst1, ComputeRegion dst2, ComputeRegion tmp)
{
	uint startIndex = (dev_heapSize / 2) * src.start;
	uint headIndex = startIndex + src.offset + roundToNextMultipleOf(src.lastVar - src.firstVar + 1) * (firstRel - 1) + i;
	uint index = graphGet(headIndex);
	while (index != NIL) {
		index += startIndex;
		uint myBits = graphGet(index + threadIdx.x);
		uint base = getValAtThread(myBits, BASE);
		uint s = decode(thirdRel, secondRel, myBits, base, i, p, dst1, dst2, tmp);
		if (s == -1) return -1;
		index = getValAtThread(myBits, NEXT);
	}
	return 0;
}

/*
__global__ void addEdges(uint* keys, uint* valIndex, const uint numKeys, uint* val1, uint* val2, uint firstVar, uint lastVar, int start, uint offset) {
__shared__ uint temp[THREADS_PER_BLOCK / WARP_SIZE * 64];
uint* p = &temp[threadIdx.y * 64];
uint startIndex = (dev_heapSize / 2) * start;
uint virtualNumVars = roundToNextMultipleOf(lastVar - firstVar + 1);
uint i = getAndIncrement(1);
while (i < numKeys) {
uint src = keys[i];
uint begin = valIndex[i];
uint end = valIndex[i + 1];
uint virtualBegin = roundToPrevMultipleOf(begin); // to ensure alignment
for (int j = virtualBegin; j < end; j += WARP_SIZE) {
uint myIndex = j + threadIdx.x;
p[threadIdx.x] = myIndex < end ? val1[myIndex] : NIL;
p[threadIdx.x + 32] = myIndex < end ? val2[myIndex] : NIL;
uint beginK = max((int)begin - j, 0);
uint endK = min(end - j, WARP_SIZE);
for (int k = beginK; k < endK; k++) {
uint dst = p[k];
uint rel = p[k + 32];
uint headIndex = startIndex + offset + virtualNumVars * (rel - 1) + src - firstVar;
uint index = getIndex(headIndex, start);
if (index == -1) {
break;
}
uint s = insert(index + startIndex, dst, start);
if (s == -1) {
break;
}
}
}
i = getAndIncrement(1);
}
resetWorklistIndex();
}
*/

// for complete rules
__global__ void compute11(ComputeRegion src, ComputeRegion dst1, ComputeRegion dst2, ComputeRegion tmp)
{
	__shared__ uint temp[THREADS_PER_BLOCK / ELEMENT_WIDTH * 32];
	uint *p = &temp[threadIdx.y * 32];
	uint numPartialVars = src.lastVar - src.firstVar + 1;
	uint i = getAndIncrement(1);
	while (i < numPartialVars) {
		for (int j = 0; j < dev_numRules; j++) {
			if (dev_rules[j * 3 + 1] != 0 && dev_rules[j * 3 + 2] != 0) {
				uint s = apply(dev_rules[j * 3 + 1], dev_rules[j * 3 + 2], dev_rules[j * 3], i, p, src, dst1, dst2, tmp);
				if (s == -1) break;
			}
		}
		i = getAndIncrement(1);
	}
	resetWorklistIndex();
}

// for rules which have two labels
__global__ void compute10(uint firstVar, uint lastVar, uint fromOffset, int start, ComputeRegion tmp)
{
	uint startIndex = (dev_heapSize / 2) * start;
	uint numPartialVars = lastVar - firstVar + 1;
	uint virtualNumPartialVars = roundToNextMultipleOf(numPartialVars);
	uint i = getAndIncrement(1);
	while (i < numPartialVars) {
		for (int j = 0; j < dev_numRules; j++) {
			if (dev_rules[j * 3 + 1] != 0 && dev_rules[j * 3 + 2] == 0) {
				uint fromHeadIndex = startIndex + fromOffset + virtualNumPartialVars * (dev_rules[j * 3 + 1] - 1) + i;
				uint fromIndex = graphGet(fromHeadIndex);
				if (fromIndex != NIL) {
					uint s = union2(i, dev_rules[j * 3], tmp, fromIndex + startIndex, start);
					if (s == -1) break;
				}
			}
		}
		i = getAndIncrement(1);
	}
	resetWorklistIndex();
}

// for rules which have only one label
__global__ void compute00(uint firstVar, uint lastVar, uint tmpOffset, int start)
{
	uint startIndex = (dev_heapSize / 2) * start;
	uint numPartialVars = lastVar - firstVar + 1;
	uint virtualNumPartialVars = roundToNextMultipleOf(numPartialVars);
	uint i = getAndIncrement(1);
	while (i < numPartialVars) {
		for (int j = 0; j < dev_numRules; j++) {
			if (dev_rules[j * 3 + 1] == 0) {
				uint headIndex = startIndex + tmpOffset + virtualNumPartialVars * (dev_rules[j * 3] - 1) + i;
				uint index = getIndex(headIndex, start);
				if (index == -1) break;
				uint s = insert(index + startIndex, firstVar + i, start);
				if (s == -1) break;
			}
		}
		i = getAndIncrement(1);
	}
	resetWorklistIndex();
}

void spagpu_s(Partition &p, int start, uint heapSize, uint *elementPool, bool &r)
{
	std::cout << "Self-matching..." << std::flush;
	uint blocks = getBlocks();
	dim3 threads(ELEMENT_WIDTH, THREADS_PER_BLOCK / ELEMENT_WIDTH);
	if (p.tmpSize == 0) {
		uint numPartialVars = p.lastVar - p.firstVar + 1;
		uint virtualNumPartialVars = roundToNextMultipleOf(numPartialVars);
		uint headSize = virtualNumPartialVars * numLabels;
		uint offset = p.oldSize + p.deltaSize;
		if (offset + headSize > heapSize / 2) { // if the size of the partition exceeds the limit, return and repart
			r = true;
			std::cout << "Need Repartition." << std::endl;
			return;
		}
		initialize(headSize, start, offset, heapSize, elementPool);
	}
	ComputeRegion empty(0, 0, 0, 0, false);
	ComputeRegion tmp_s(p.firstVar, p.lastVar, start, p.oldSize + p.deltaSize, true);
	if (p.oldSize == 0) {
		compute00<<<blocks, threads>>>(p.firstVar, p.lastVar, p.oldSize + p.deltaSize, start);
		needRepartition(p, heapSize, r);
		if (r) return;
	}
	compute10<<<blocks, threads>>>(p.firstVar, p.lastVar, p.oldSize, start, tmp_s);
	needRepartition(p, heapSize, r);
	if (r) return;
	ComputeRegion new_s(p.firstVar, p.lastVar, start, p.oldSize, true);
	if (p.oldSize != 0) {
		ComputeRegion old_s(p.firstVar, p.lastVar, start, 0, true);
		compute11<<<blocks, threads>>>(old_s, new_s, empty, tmp_s);
		needRepartition(p, heapSize, r);
		if (r) return;
		compute11<<<blocks, threads>>>(new_s, old_s, new_s, tmp_s);
		needRepartition(p, heapSize, r);
		if (r) return;
	} else {
		compute11<<<blocks, threads>>>(new_s, new_s, empty, tmp_s);
		needRepartition(p, heapSize, r);
		if (r) return;
	}
	uint poolSize;
	if (start) {
		cudaSafeCall(cudaMemcpyFromSymbol(&poolSize, freeList2, sizeof(uint)));
	} else {
		cudaSafeCall(cudaMemcpyFromSymbol(&poolSize, freeList1, sizeof(uint)));
	}
	p.tmpSize = poolSize - p.deltaSize - p.oldSize;
	std::cout << "OK." << std::endl;
}

void spagpu_b(Partition &p1, Partition &p2, bool &r1, bool &r2, uint heapSize, uint *elementPool)
{
	uint blocks = getBlocks();
	dim3 threads(ELEMENT_WIDTH, THREADS_PER_BLOCK / ELEMENT_WIDTH);
	if (p1.tmpSize == 0) {
		uint numPartialVars1 = p1.lastVar - p1.firstVar + 1;
		uint virtualNumPartialVars1 = roundToNextMultipleOf(numPartialVars1);
		uint headSize1 = virtualNumPartialVars1 * numLabels;
		uint offset1 = p1.oldSize + p1.deltaSize;
		if (offset1 + headSize1 > heapSize / 2) {
			r1 = true;
			std::cout << "Need Repartition." << std::endl;
			return;
		}
		initialize(headSize1, 0, offset1, heapSize, elementPool);
	}
	if (p2.tmpSize == 0) {
		uint numPartialVars2 = p2.lastVar - p2.firstVar + 1;
		uint virtualNumPartialVars2 = roundToNextMultipleOf(numPartialVars2);
		uint headSize2 = virtualNumPartialVars2 * numLabels;
		uint offset2 = p2.oldSize + p2.deltaSize;
		if (offset2 + headSize2 > heapSize / 2) {
			r2 = true;
			std::cout << "Need Repartition." << std::endl;
			return;
		}
		initialize(headSize2, 1, offset2, heapSize, elementPool);
	}
	ComputeRegion empty(0, 0, 0, 0, false);
	ComputeRegion tmp1(p1.firstVar, p1.lastVar, 0, p1.oldSize + p1.deltaSize, true);
	ComputeRegion tmp2(p2.firstVar, p2.lastVar, 1, p2.oldSize + p2.deltaSize, true);
	std::cout << "## ITERATION 0 ##" << std::endl;
	if (p1.oldSize != 0 && p2.deltaSize != 0) {
		ComputeRegion old1(p1.firstVar, p1.lastVar, 0, 0, true);
		ComputeRegion new2(p2.firstVar, p2.lastVar, 1, p2.oldSize, true);
		compute11<<<blocks, threads>>>(old1, new2, empty, tmp1);
		needRepartition(p1, heapSize, r1);
		if (r1) return;
	}
	if (p1.deltaSize != 0) {
		ComputeRegion new1(p1.firstVar, p1.lastVar, 0, p1.oldSize, true);
		if (p2.oldSize != 0 && p2.deltaSize != 0) {
			ComputeRegion old2(p2.firstVar, p2.lastVar, 1, 0, true);
			ComputeRegion new2(p2.firstVar, p2.lastVar, 1, p2.oldSize, true);
			compute11<<<blocks, threads>>>(new1, old2, new2, tmp1);
			needRepartition(p1, heapSize, r1);
			if (r1) return;
		}
		else {
			if (p2.oldSize != 0) {
				ComputeRegion old2(p2.firstVar, p2.lastVar, 1, 0, true);
				compute11<<<blocks, threads>>>(new1, old2, empty, tmp1);
				needRepartition(p1, heapSize, r1);
				if (r1) return;
			}
			if (p2.deltaSize != 0) {
				ComputeRegion new2(p2.firstVar, p2.lastVar, 1, p2.oldSize, true);
				compute11<<<blocks, threads>>>(new1, new2, empty, tmp1);
				needRepartition(p1, heapSize, r1);
				if (r1) return;
			}
		}
	}
	uint poolSize1;
	cudaSafeCall(cudaMemcpyFromSymbol(&poolSize1, freeList1, sizeof(uint)));
	p1.tmpSize = poolSize1 - p1.deltaSize - p1.oldSize;
	if (p2.oldSize != 0 && p1.deltaSize != 0) {
		ComputeRegion old2(p2.firstVar, p2.lastVar, 1, 0, true);
		ComputeRegion new1(p1.firstVar, p1.lastVar, 0, p1.oldSize, true);
		compute11<<<blocks, threads>>>(old2, new1, empty, tmp2);
		needRepartition(p2, heapSize, r2);
		if (r2) return;
	}
	if (p2.deltaSize != 0) {
		ComputeRegion new2(p2.firstVar, p2.lastVar, 1, p2.oldSize, true);
		if (p1.oldSize != 0 && p1.deltaSize != 0) {
			ComputeRegion old1(p1.firstVar, p1.lastVar, 0, 0, true);
			ComputeRegion new1(p1.firstVar, p1.lastVar, 0, p1.oldSize, true);
			compute11<<<blocks, threads>>>(new2, old1, new1, tmp2);
			needRepartition(p2, heapSize, r2);
			if (r2) return;
		}
		else {
			if (p1.oldSize != 0) {
				ComputeRegion old1(p1.firstVar, p1.lastVar, 0, 0, true);
				compute11<<<blocks, threads>>>(new2, old1, empty, tmp2);
				needRepartition(p2, heapSize, r2);
				if (r2) return;
			}
			if (p1.deltaSize != 0) {
				ComputeRegion new1(p1.firstVar, p1.lastVar, 0, p1.oldSize, true);
				compute11<<<blocks, threads>>>(new2, new1, empty, tmp2);
				needRepartition(p2, heapSize, r2);
				if (r2) return;
			}
		}
	}
	uint poolSize2;
	cudaSafeCall(cudaMemcpyFromSymbol(&poolSize2, freeList2, sizeof(uint)));
	p2.tmpSize = poolSize2 - p2.deltaSize - p2.oldSize;
}

void spagpu(Partition &p1, Partition &p2, bool &r1, bool &r2, uint heapSize)
{
	uint blocks = getBlocks();
	dim3 threads(ELEMENT_WIDTH, THREADS_PER_BLOCK / ELEMENT_WIDTH);
	ComputeRegion empty(0, 0, 0, 0, false);
	ComputeRegion tmp1(p1.firstVar, p1.lastVar, 0, p1.oldSize + p1.deltaSize, true);
	ComputeRegion tmp2(p2.firstVar, p2.lastVar, 1, p2.oldSize + p2.deltaSize, true);
	//  repeat until a ﬁxed point is reached
	int iterNo = 0;
	for (;;) {
		std::cout << "## ITERATION " << ++iterNo << " ##" << std::endl;
		bool changed = false;
		cudaSafeCall(cudaMemcpyToSymbol(dev_changed, &changed, sizeof(bool)));
		if (p1.oldSize != 0) {
			ComputeRegion old1(p1.firstVar, p1.lastVar, 0, 0, true);
			compute11<<<blocks, threads>>>(old1, tmp1, tmp2, tmp1);
			needRepartition(p1, heapSize, r1);
			if (r1) return;
		}
		if (p1.deltaSize != 0) {
			ComputeRegion new1(p1.firstVar, p1.lastVar, 0, p1.oldSize, true);
			compute11<<<blocks, threads>>>(new1, tmp1, tmp2, tmp1);
			needRepartition(p1, heapSize, r1);
			if (r1) return;
		}
		if (p1.oldSize != 0 && p1.deltaSize != 0) {
			ComputeRegion old1(p1.firstVar, p1.lastVar, 0, 0, true);
			ComputeRegion new1(p1.firstVar, p1.lastVar, 0, p1.oldSize, true);
			compute11<<<blocks, threads>>>(tmp1, old1, new1, tmp1);
			needRepartition(p1, heapSize, r1);
			if (r1) return;
		} else {
			if (p1.oldSize != 0) {
				ComputeRegion old1(p1.firstVar, p1.lastVar, 0, 0, true);
				compute11<<<blocks, threads>>>(tmp1, old1, empty, tmp1);
				needRepartition(p1, heapSize, r1);
				if (r1) return;
			}
			if (p1.deltaSize != 0) {
				ComputeRegion new1(p1.firstVar, p1.lastVar, 0, p1.oldSize, true);
				compute11<<<blocks, threads>>>(tmp1, new1, empty, tmp1);
				needRepartition(p1, heapSize, r1);
				if (r1) return;
			}
		}
		if (p2.oldSize != 0 && p2.deltaSize != 0) {
			ComputeRegion old2(p2.firstVar, p2.lastVar, 1, 0, true);
			ComputeRegion new2(p2.firstVar, p2.lastVar, 1, p2.oldSize, true);
			compute11<<<blocks, threads>>>(tmp1, old2, new2, tmp1);
			needRepartition(p1, heapSize, r1);
			if (r1) return;
		} else {
			if (p2.oldSize != 0) {
				ComputeRegion old2(p2.firstVar, p2.lastVar, 1, 0, true);
				compute11<<<blocks, threads>>>(tmp1, old2, empty, tmp1);
				needRepartition(p1, heapSize, r1);
				if (r1) return;
			}
			if (p2.deltaSize != 0) {
				ComputeRegion new2(p2.firstVar, p2.lastVar, 1, p2.oldSize, true);
				compute11<<<blocks, threads>>>(tmp1, new2, empty, tmp1);
				needRepartition(p1, heapSize, r1);
				if (r1) return;
			}
		}
		compute10<<<blocks, threads>>>(p1.firstVar, p1.lastVar, p1.oldSize + p1.deltaSize, 0, tmp1);
		needRepartition(p1, heapSize, r1);
		if (r1) return;
		compute11<<<blocks, threads>>>(tmp1, tmp1, tmp2, tmp1);
		needRepartition(p1, heapSize, r1);
		if (r1) return;
		uint poolSize1;
		cudaSafeCall(cudaMemcpyFromSymbol(&poolSize1, freeList1, sizeof(uint)));
		p1.tmpSize = poolSize1 - p1.deltaSize - p1.oldSize;
		if (p2.oldSize != 0) {
			ComputeRegion old2(p2.firstVar, p2.lastVar, 1, 0, true);
			compute11<<<blocks, threads>>>(old2, tmp1, tmp2, tmp2);
			needRepartition(p2, heapSize, r2);
			if (r2) return;
		}
		if (p2.deltaSize != 0) {
			ComputeRegion new2(p2.firstVar, p2.lastVar, 1, p2.oldSize, true);
			compute11<<<blocks, threads>>>(new2, tmp1, tmp2, tmp2);
			needRepartition(p2, heapSize, r2);
			if (r2) return;
		}
		if (p2.oldSize != 0 && p2.deltaSize != 0) {
			ComputeRegion old2(p2.firstVar, p2.lastVar, 1, 0, true);
			ComputeRegion new2(p2.firstVar, p2.lastVar, 1, p2.oldSize, true);
			compute11<<<blocks, threads>>>(tmp2, old2, new2, tmp2);
			needRepartition(p2, heapSize, r2);
			if (r2) return;
		} else {
			if (p2.oldSize != 0) {
				ComputeRegion old2(p2.firstVar, p2.lastVar, 1, 0, true);
				compute11<<<blocks, threads>>>(tmp2, old2, empty, tmp2);
				needRepartition(p2, heapSize, r2);
				if (r2) return;
			}
			if (p2.deltaSize != 0) {
				ComputeRegion new2(p2.firstVar, p2.lastVar, 1, p2.oldSize, true);
				compute11<<<blocks, threads>>>(tmp2, new2, empty, tmp2);
				needRepartition(p2, heapSize, r2);
				if (r2) return;
			}
		}
		if (p1.oldSize != 0 && p1.deltaSize != 0) {
			ComputeRegion old1(p1.firstVar, p1.lastVar, 0, 0, true);
			ComputeRegion new1(p1.firstVar, p1.lastVar, 0, p1.oldSize, true);
			compute11<<<blocks, threads>>>(tmp2, old1, new1, tmp2);
			needRepartition(p2, heapSize, r2);
			if (r2) return;
		} else {
			if (p1.oldSize != 0) {
				ComputeRegion old1(p1.firstVar, p1.lastVar, 0, 0, true);
				compute11<<<blocks, threads>>>(tmp2, old1, empty, tmp2);
				needRepartition(p2, heapSize, r2);
				if (r2) return;
			}
			if (p1.deltaSize != 0) {
				ComputeRegion new1(p1.firstVar, p1.lastVar, 0, p1.oldSize, true);
				compute11<<<blocks, threads>>>(tmp2, new1, empty, tmp2);
				needRepartition(p2, heapSize, r2);
				if (r2) return;
			}
		}
		compute10<<<blocks, threads>>>(p2.firstVar, p2.lastVar, p2.oldSize + p2.deltaSize, 1, tmp2);
		needRepartition(p2, heapSize, r2);
		if (r2) return;
		compute11<<<blocks, threads>>>(tmp2, tmp1, tmp2, tmp2);
		needRepartition(p2, heapSize, r2);
		if (r2) return;
		uint poolSize2;
		cudaSafeCall(cudaMemcpyFromSymbol(&poolSize2, freeList2, sizeof(uint)));
		p2.tmpSize = poolSize2 - p2.deltaSize - p2.oldSize;
		cudaSafeCall(cudaMemcpyFromSymbol(&changed, dev_changed, sizeof(bool)));
		if (changed == false) break;
	}
}

__device__ void computeDegreePerLabel(uint *degree_elements, uint *degree_edges, uint i, uint index, uint startIndex, uint *p)
{
	do {
		if (isFirstThreadOfVWarp()) degree_elements[i]++;
		index += startIndex;
		uint myBits = graphGet(index + threadIdx.x);
		p[threadIdx.x] = threadIdx.x < BASE ? __popc(myBits) : 0;
		int k = blockDim.x / 2;
		while (k) {
			if (threadIdx.x < k) p[threadIdx.x] += p[threadIdx.x + k];
			k /= 2;
		}
		if (isFirstThreadOfVWarp()) degree_edges[i] += p[0];
		index = getValAtThread(myBits, NEXT);
	} while (index != NIL);
}

__global__ void computeDegree(uint *degree_elements, uint *degree_edges, uint numPartialVars, int start, uint offset)
{
	__shared__ uint temp[THREADS_PER_BLOCK];
	uint *p = &temp[threadIdx.y * blockDim.x];
	uint startIndex = (dev_heapSize / 2) * start;
	uint virtualNumPartialVars = roundToNextMultipleOf(numPartialVars);
	uint i = getAndIncrement(1);
	while (i < numPartialVars) {
		for (int j = 0; j < dev_numLabels; j++) {
			uint headIndex = startIndex + offset + virtualNumPartialVars * j + i;
			uint index = graphGet(headIndex);
			if (index != NIL)
				computeDegreePerLabel(degree_elements, degree_edges, i, index, startIndex, p);
		}
		i = getAndIncrement(1);
	}
	resetWorklistIndex();
}

void getDegree(Partition p, int start, uint *degree)
{
	uint numPartialVars = p.lastVar - p.firstVar + 1;
	uint *host_degree = new uint[numPartialVars * 6]();
	uint *dev_degree;
	size_t size = numPartialVars * 6 * sizeof(uint);
	cudaSafeCall(cudaMalloc((void **)&dev_degree, size));
	cudaSafeCall(cudaMemset(dev_degree, 0, size));
	uint blocks = getBlocks();
	dim3 threads(ELEMENT_WIDTH, THREADS_PER_BLOCK / ELEMENT_WIDTH);
	if (p.oldSize != 0)
		computeDegree<<<blocks, threads>>>(dev_degree + numPartialVars * 3, dev_degree, numPartialVars, start, 0);
	if (p.deltaSize != 0)
		computeDegree<<<blocks, threads>>>(dev_degree + numPartialVars * 4, dev_degree + numPartialVars, numPartialVars, start, p.oldSize);
	if (p.tmpSize != 0)
		computeDegree<<<blocks, threads>>>(dev_degree + numPartialVars * 5, dev_degree + numPartialVars * 2, numPartialVars, start, p.oldSize + p.deltaSize);
	cudaSafeCall(cudaMemcpy(host_degree, dev_degree, size, D2H));
	cudaFree(dev_degree);
	for (int i = 0; i < 6; i++)
		memcpy(degree + p.firstVar + numVars * i, host_degree + numPartialVars * i, numPartialVars * sizeof(uint));
	delete[] host_degree;
}

__global__ void merge(ComputeRegion old, uint fromOffset, int fromStart)
{
	uint numPartialVars = old.lastVar - old.firstVar + 1;
	uint virtualNumPartialVars = roundToNextMultipleOf(numPartialVars);
	uint fromStartIndex = (dev_heapSize / 2) * fromStart;
	uint i = getAndIncrement(1);
	while (i < numPartialVars) {
		for (int j = 0; j < dev_numLabels; j++) {
			uint fromHeadIndex = fromStartIndex + fromOffset + virtualNumPartialVars * j + i;
			uint fromIndex = graphGet(fromHeadIndex);
			if (fromIndex != NIL)
				union2(i, j + 1, old, fromIndex + fromStartIndex, fromStart);
		}
		i = getAndIncrement(1);
	}
	resetWorklistIndex();
}

__device__ bool ballot(uint myBits)
{
	__shared__ volatile bool temp[THREADS_PER_BLOCK / ELEMENT_WIDTH];
	if (isFirstThreadOfVWarp())
		temp[threadIdx.y] = false;
	if (threadIdx.x < BASE && myBits != 0)
		temp[threadIdx.y] = true;
	return temp[threadIdx.y];
}

__device__ void removeDuplicates(uint toHeadIndex, uint subHeadIndex, uint myBase, uint myBits, int toStart)
{
	uint toStartIndex = (dev_heapSize / 2) * toStart;
	uint subIndex = graphGet(subHeadIndex);
	if (subIndex == NIL) {
		uint toIndex = getIndex(toHeadIndex, toStart);
		addElement(toIndex + toStartIndex, myBase, myBits, toStart);
		return;
	}
	subIndex += toStartIndex;
	uint subBits = graphGet(subIndex + threadIdx.x);
	uint subBase = getValAtThread(subBits, BASE);
	uint subNext = getValAtThread(subBits, NEXT);
	for (;;) {
		if (subBase > myBase) {
			uint toIndex = getIndex(toHeadIndex, toStart);
			addElement(toIndex + toStartIndex, myBase, myBits, toStart);
			return;
		} else if (subBase == myBase) {
			if (threadIdx.x < BASE)
				myBits &= ~subBits;
			bool nonEmpty = ballot(myBits);
			if (nonEmpty) {
				uint toIndex = getIndex(toHeadIndex, toStart);
				addElement(toIndex + toStartIndex, myBase, myBits, toStart);
			}
			return;
		} else {
			if (subNext == NIL) {
				uint toIndex = getIndex(toHeadIndex, toStart);
				addElement(toIndex + toStartIndex, myBase, myBits, toStart);
				return;
			}
			subIndex = subNext + toStartIndex;
			subBits = graphGet(subIndex + threadIdx.x);
			subBase = getValAtThread(subBits, BASE);
			subNext = getValAtThread(subBits, NEXT);
		}
	}
}

__device__ void computeDiff(uint toHeadIndex, uint fromIndex, uint subHeadIndex, int toStart, int fromStart)
{
	uint fromStartIndex = (dev_heapSize / 2) * fromStart;
	do {
		fromIndex += fromStartIndex;
		uint myBits = graphGet(fromIndex + threadIdx.x);
		uint myBase = getValAtThread(myBits, BASE);
		fromIndex = getValAtThread(myBits, NEXT);
		if (threadIdx.x == NEXT) myBits = NIL;
		removeDuplicates(toHeadIndex, subHeadIndex, myBase, myBits, toStart);
	} while (fromIndex != NIL);
}

__global__ void diff(uint numPartialVars, uint toOffset, int toStart, uint fromOffset, int fromStart)
{
	uint virtualNumPartialVars = roundToNextMultipleOf(numPartialVars);
	uint fromStartIndex = (dev_heapSize / 2) * fromStart;
	uint toStartIndex = (dev_heapSize / 2) * toStart;
	uint i = getAndIncrement(1);
	while (i < numPartialVars) {
		for (int j = 0; j < dev_numLabels; j++) {
			uint fromHeadIndex = fromStartIndex + fromOffset + virtualNumPartialVars * j + i;
			uint fromIndex = graphGet(fromHeadIndex);
			if (fromIndex == NIL) continue;
			uint subHeadIndex = toStartIndex + virtualNumPartialVars * j + i;
			uint toHeadIndex = toStartIndex + toOffset + virtualNumPartialVars * j + i;
			computeDiff(toHeadIndex, fromIndex, subHeadIndex, toStart, fromStart);
		}
		i = getAndIncrement(1);
	}
	resetWorklistIndex();
}

void mergeAndDiff(Partition &p, uint heapSize, uint *elementPool)
{
	std::cout << "Updating..." << std::flush;
	uint blocks = getBlocks();
	dim3 threads(ELEMENT_WIDTH, THREADS_PER_BLOCK / ELEMENT_WIDTH);
	uint oldSize = p.oldSize;
	uint newSize = p.deltaSize;
	uint tmpSize = p.tmpSize;
	if (newSize != 0) {
		if (oldSize == 0) {
			p.oldSize = newSize;
			p.deltaSize = 0;
		} else {
			cudaSafeCall(cudaMemcpy(elementPool + heapSize / 2 + oldSize, elementPool + oldSize, newSize * sizeof(uint), D2D));
			uint poolSize = oldSize;
			cudaSafeCall(cudaMemcpyToSymbol(freeList1, &poolSize, sizeof(uint)));
			ComputeRegion old(p.firstVar, p.lastVar, 0, 0, true);
			merge<<<blocks, threads>>>(old, oldSize, 1);
			cudaSafeCall(cudaMemcpyFromSymbol(&poolSize, freeList1, sizeof(uint)));
			p.oldSize = poolSize;
			p.deltaSize = 0;
		}
	}
	if (tmpSize != 0) {
		uint fromOffset = oldSize + newSize;
		cudaSafeCall(cudaMemcpy(elementPool + heapSize / 2 + fromOffset, elementPool + fromOffset, tmpSize * sizeof(uint), D2D));
		uint numPartialVars = p.lastVar - p.firstVar + 1;
		uint virtualNumPartialVars = roundToNextMultipleOf(numPartialVars);
		uint headSize = virtualNumPartialVars * numLabels;
		initialize(headSize, 0, p.oldSize, heapSize, elementPool);
		diff<<<blocks, threads>>>(numPartialVars, p.oldSize, 0, fromOffset, 1);
		uint poolSize;
		cudaSafeCall(cudaMemcpyFromSymbol(&poolSize, freeList1, sizeof(uint)));
		if (poolSize - p.oldSize == headSize) {
			poolSize = p.oldSize;
			cudaSafeCall(cudaMemcpyToSymbol(freeList1, &poolSize, sizeof(uint)));
		} else {
			p.deltaSize = poolSize - p.oldSize;
		}
		p.tmpSize = 0;
	}
	std::cout << "OK." << std::endl;
}

