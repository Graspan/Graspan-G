//
// main.cpp
// spagpu
//
// Created by LuS on 2018/3/13.
//

#include <map>
#include <fstream>
#include <sstream>
#include <utility>
#include <set>
#include <vector>

#include <mutex>
#include <condition_variable>
#include <boost/asio/io_service.hpp>
#include <boost/thread/thread.hpp>
#include <boost/bind.hpp>

#include "kernel.h"

using namespace std;

mutex addEdges_mtx, comp_mtx;
condition_variable cv;
uint numFinished;
bool finished;

void checkGPUConfiguration()
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	cout << "NUM COMPUTE-CAPABLE DEVICES: " << deviceCount << endl;
	if (deviceCount == 0) {
		cerr << "There is no device supporting CUDA." << endl;
		exit(-1);
	}
	selectedDevice = 0;
	if (deviceCount > 1) {
		int maxMultiprocessors = 0;
		for (int device = 0; device < deviceCount; device++) {
			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, device);
			if (maxMultiprocessors < deviceProp.multiProcessorCount) {
				maxMultiprocessors = deviceProp.multiProcessorCount;
				selectedDevice = device;
			}
		}
		cudaSetDevice(selectedDevice);
	}
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, selectedDevice);
	cout << "SELECTED DEVICE: " << selectedDevice << " " << deviceProp.name << endl;
	size_t curAvailDeviceMemoryInBytes, totalDeviceMemoryInBytes;
	cudaMemGetInfo(&curAvailDeviceMemoryInBytes, &totalDeviceMemoryInBytes);
	cout << "GPU's total memory: " << B2MB(totalDeviceMemoryInBytes) << " MB, free memory: " << B2MB(curAvailDeviceMemoryInBytes) << " MB" << endl;
}

string skipComments(ifstream &fin)
{
	string line;
	for (;;) {
		getline(fin, line);
		if (line[0] != '#') return line;
	}
}

void readAndTransferRules(char *filePath, map<string, int> &labelMap)
{
	ifstream fin;
	fin.open(filePath);
	string line = skipComments(fin);
	int numRules = atoi(line.c_str());
	cout << "NUM RULES: " << numRules << endl;
	int *rules = new int[numRules * 3]();
	int k = 1;
	for (int i = 0; i < numRules; i++) {
		string line = skipComments(fin);
		istringstream lineStream(line);
		int j = 0;
		string item;
		while (getline(lineStream, item, '\t')) {
			if (labelMap.count(item) == 0) {
				labelMap.insert(pair<string, int>(item, k));
				k++;
			}
			rules[i * 3 + j] = labelMap[item];
			j++;
		}
	}
	numLabels = labelMap.size();
	cout << "NUM LABELS: " << numLabels << endl;
	fin.close();
	transferRules(numRules, rules);
	delete[] rules;
}

string nextItem(istringstream &lineStream)
{
	string item;
	getline(lineStream, item, '\t');
	return item;
}

uint* getGraphInfo(char *filePath, map<string, int> labelMap)
{
	ifstream fin;
	fin.open(filePath);
	string line = skipComments(fin);
	istringstream lineStream(line);
	uint numEdges = atoi(nextItem(lineStream).c_str());
	numVars = atoi(nextItem(lineStream).c_str());
	cout << "NUM VERTICES: " << numVars << endl;
	cout << "INITIAL NUM EDGES: " << numEdges << endl;
	// num of old/delta/tmp edges/elements per vertex
	uint *degree = new uint[numVars * 6]();
	set<pair<uint, int>> s;
	uint tmp = 0;
	for (uint i = 0; i < numEdges; i++) {
		string line = skipComments(fin);
		istringstream lineStream(line);
		uint src = atoi(nextItem(lineStream).c_str());
		degree[src + numVars]++;
		if (src != tmp) {
			degree[tmp + numVars * 4] = s.size();
			s.clear();
			tmp = src;
		}
		uint dst = atoi(nextItem(lineStream).c_str());
		uint base = BASE_OF(dst);
		int rel = labelMap[nextItem(lineStream)];
		s.insert(pair<uint, int>(base, rel));
	}
	degree[tmp + numVars * 4] = s.size();
	s.clear();
	fin.close();
	return degree;
}

void initializePartitions(vector<Partition> &partitions, uint *degree, uint heapSize)
{
	uint partitionSize = heapSize / 4;
	uint tmp1 = 0;
	uint tmp2 = 0;
	uint tmp3 = 0;
	for (uint i = 0; i < numVars; i++) {
		tmp1 += degree[i + numVars * 4];
		tmp2 += degree[i + numVars];
		if (tmp1 * ELEMENT_WIDTH + roundToNextMultipleOf(i - tmp3 + 1) * numLabels > partitionSize) {
			partitions.push_back(Partition(tmp3, i - 1, 0, tmp2 - degree[i + numVars], 0, false));
			tmp1 = degree[i + numVars * 4];
			tmp2 = degree[i + numVars];
			tmp3 = i;
		}
	}
	partitions.push_back(Partition(tmp3, numVars - 1, 0, tmp2, 0, false));
	// for small graphs
	if (partitions.size() == 1) {
		uint tmp4 = 0;
		uint tmp5 = 0;
		for (uint i = 0; i < numVars; i++) {
			tmp4 += degree[i + numVars * 4];
			tmp5 += degree[i + numVars];
			if (tmp4 > tmp1 / 2) {
				partitions[0].lastVar = i - 1;
				partitions[0].deltaSize = tmp5 - degree[i + numVars];
				tmp3 = i;
				break;
			}
		}
		partitions.push_back(Partition(tmp3, numVars - 1, 0, tmp2 - partitions[0].deltaSize, 0, false));
	}
	cout << "INITIAL NUM PARTITIONS: " << partitions.size() << endl;
}

uint* readEdges(char *filePath, map<string, int> labelMap, Partition p, uint &numKeys, uint* &valIndex)
{
	cout << "Reading Edges..." << flush;
	ifstream fin;
	fin.open(filePath);
	string line = skipComments(fin);
	istringstream lineStream(line);
	uint numEdges = atoi(nextItem(lineStream).c_str());
	nextItem(lineStream);
	uint *edges = new uint[p.deltaSize * 3];
	uint k = 0;
	for (uint i = 0; i < numEdges; i++) {
		string line = skipComments(fin);
		istringstream lineStream(line);
		uint src = atoi(nextItem(lineStream).c_str());
		if (src < p.firstVar) {
			nextItem(lineStream);
			nextItem(lineStream);
		} else if (src >= p.firstVar && src <= p.lastVar) {
			edges[k] = src;
			uint dst = atoi(nextItem(lineStream).c_str());
			edges[k + p.deltaSize] = dst;
			uint rel = labelMap[nextItem(lineStream)];
			edges[k + p.deltaSize * 2] = rel;
			k++;
		} else {
			break;
		}
	}
	fin.close();
	valIndex = new uint[p.deltaSize + 1];
	valIndex[0] = 0;
	k = 1;
	for (uint i = 1; i < p.deltaSize; i++) {
		if (edges[i] != edges[i - 1]) {
			valIndex[k] = i;
			edges[k] = edges[i];
			k++;
		}
	}
	numKeys = k;
	valIndex[k] = p.deltaSize;
	cout << "OK." << endl;
	return edges;
}

void insert(uint index, uint var, uint *elements, uint &freeList)
{
	uint base = BASE_OF(var);
	uint unit = UNIT_OF(var);
	uint bit = BIT_OF(var);
	uint myBits = 1 << bit;
	for (;;) {
		uint toBase = elements[index + BASE];
		if (toBase == NIL) {
			elements[index + BASE] = base;
			elements[index + unit] = myBits;
			elements[index + NEXT] = NIL;
			return;
		}
		if (toBase == base) {
			elements[index + unit] |= myBits;
			return;
		}
		if (toBase < base) {
			uint toNext = elements[index + NEXT];
			if (toNext == NIL) {
				uint newIndex = freeList;
				freeList += ELEMENT_WIDTH;
				elements[newIndex + BASE] = base;
				elements[newIndex + unit] = myBits;
				elements[newIndex + NEXT] = NIL;
				elements[index + NEXT] = newIndex;
				return;
			}
			index = toNext;
		} else {
			uint newIndex = freeList;
			freeList += ELEMENT_WIDTH;
			memcpy(elements + newIndex, elements + index, ELEMENT_WIDTH * sizeof(uint));
			memset(elements + index, 0, ELEMENT_WIDTH * sizeof(uint));
			elements[index + BASE] = base;
			elements[index + unit] = myBits;
			elements[index + NEXT] = newIndex;
			return;
		}
	}
}

void initializeEdges(uint lower, uint upper, Partition p, uint *edges, uint *valIndex, uint *elements, uint freeList, uint numTasks)
{
	uint numPartialVars = p.lastVar - p.firstVar + 1;
	uint virtualNumPartialVars = roundToNextMultipleOf(numPartialVars);
	for (uint i = lower; i < upper; i++) {
		uint src = edges[i];
		uint begin = valIndex[i];
		uint end = valIndex[i + 1];
		for (uint j = begin; j < end; j++) {
			uint dst = edges[p.deltaSize + j];
			uint rel = edges[p.deltaSize * 2 + j];
			uint headIndex = virtualNumPartialVars * (rel - 1) + src - p.firstVar;
			uint index = elements[headIndex];
			if (index == NIL) {
				index = freeList;
				freeList += ELEMENT_WIDTH;
				elements[index + BASE] = NIL;
				elements[headIndex] = index;
			}
			insert(index, dst, elements, freeList);
		}
	}
	unique_lock<mutex> lck(addEdges_mtx);
	numFinished++;
	if (numFinished == numTasks) {
		finished = true;
		cv.notify_one();
	}
}

uint* createGraph(Partition &p, uint *edges, uint numKeys, uint *valIndex, uint *degree, boost::asio::io_service &ioServ)
{
	cout << "Creating Graph..." << flush;
	uint numElements = 0;
	for (uint i = p.firstVar; i <= p.lastVar; i++)
		numElements += degree[i + numVars * 4];
	uint numPartialVars = p.lastVar - p.firstVar + 1;
	uint virtualNumPartialVars = roundToNextMultipleOf(numPartialVars);
	uint headSize = virtualNumPartialVars * numLabels;
	uint poolSize = headSize + numElements * ELEMENT_WIDTH;
	uint *elements = new uint[poolSize]();
	uint segSize = numKeys / 48 + 1;
	uint nSegs = numKeys / segSize;
	if (numKeys % segSize != 0) nSegs++;
	numFinished = 0;
	finished = false;
	uint freeList = 0;
	memset(elements, -1, headSize * sizeof(uint));
	freeList += headSize;
	uint lower, upper;
	for (uint i = 0; i < nSegs; i++) {
		lower = i * segSize;
		upper = (i == nSegs - 1) ? numKeys : lower + segSize;
		ioServ.post(boost::bind(initializeEdges, lower, upper, p, edges, valIndex, elements, freeList, nSegs));
		for (uint j = lower; j < upper; j++) {
			uint src = edges[j];
			freeList += degree[src + numVars * 4] * ELEMENT_WIDTH;
		}
	}
	unique_lock<mutex> lck(comp_mtx);
	while (!finished) cv.wait(lck);
	p.deltaSize = poolSize;
	p.flag = true;
	delete[] edges;
	delete[] valIndex;
	cout << "OK." << endl;
	return elements;
}

void edges2Elements(uint lower, uint upper, Partition p, uint *elements, uint offset, uint freeList, uchar *edges, uint *degree, uint numTasks)
{
	uint numPartialVars = p.lastVar - p.firstVar + 1;
	uint virtualNumPartialVars = roundToNextMultipleOf(numPartialVars);
	for (uint i = lower; i < upper; i++) {
		uint numEdges_i = degree[p.firstVar + i];
		for (uint j = 0; j < numEdges_i; j++) {
			uint dst = *((uint*)(edges + size_t(j) * 5));
			uint rel = *(edges + size_t(j) * 5 + 4);
			uint headIndex = offset + virtualNumPartialVars * (rel - 1) + i;
			uint index = elements[headIndex];
			if (index == NIL) {
				index = freeList;
				freeList += ELEMENT_WIDTH;
				elements[index + BASE] = NIL;
				elements[headIndex] = index;
			}
			insert(index, dst, elements, freeList);
		}
		edges += size_t(numEdges_i) * 5;
	}
	unique_lock<mutex> lck(addEdges_mtx);
	numFinished++;
	if (numFinished == numTasks) {
		finished = true;
		cv.notify_one();
	}
}

uint* readElements_c(Partition p, int k, uint *degree, boost::asio::io_service &ioServ)
{
	uint numOldEdges = 0, numDeltaEdges = 0, numTmpEdges = 0;
	for (uint i = p.firstVar; i <= p.lastVar; i++) {
		numOldEdges += degree[i];
		numDeltaEdges += degree[i + numVars];
		numTmpEdges += degree[i + numVars * 2];
	}
	uint numEdges = numOldEdges + numDeltaEdges + numTmpEdges;
	size_t size = size_t(numEdges) * 5;
	uchar *edges = new uchar[size];
	string str = "Partition" + to_string(k);
	FILE *fp = fopen(str.c_str(), "rb");
	fread(edges, 1, size, fp);
	fclose(fp);
	uint poolSize = p.oldSize + p.deltaSize + p.tmpSize;
	uint *elements = new uint[poolSize]();
	uint numPartialVars = p.lastVar - p.firstVar + 1;
	uint virtualNumPartialVars = roundToNextMultipleOf(numPartialVars);
	uint headSize = virtualNumPartialVars * numLabels;
	uint segSize = numPartialVars / 16 + 1;
	uint nSegs = numPartialVars / segSize;
	if (numPartialVars % segSize != 0) nSegs++;
	uint numTasks = 0;
	if (numOldEdges != 0) numTasks += nSegs;
	if (numDeltaEdges != 0) numTasks += nSegs;
	if (numTmpEdges != 0) numTasks += nSegs;
	numFinished = 0;
	finished = false;
	uint c = 0;
	uint freeList = 0;
	uint lower, upper;
	if (p.oldSize != 0) {
		memset(elements, -1, headSize * sizeof(uint));
		freeList += headSize;
		if (numOldEdges != 0) {
			for (uint i = 0; i < nSegs; i++) {
				lower = i * segSize;
				upper = (i == nSegs - 1) ? numPartialVars : lower + segSize;
				ioServ.post(boost::bind(edges2Elements, lower, upper, p, elements, 0, freeList, edges + size_t(c) * 5, degree, numTasks));
				for (uint j = lower; j < upper; j++) {
					c += degree[p.firstVar + j];
					freeList += degree[p.firstVar + j + numVars * 3] * ELEMENT_WIDTH;
				}
			}
		}
	}
	if (p.deltaSize != 0) {
		memset(elements + p.oldSize, -1, headSize * sizeof(uint));
		freeList += headSize;
		if (numDeltaEdges != 0) {
			for (uint i = 0; i < nSegs; i++) {
				lower = i * segSize;
				upper = (i == nSegs - 1) ? numPartialVars : lower + segSize;
				ioServ.post(boost::bind(edges2Elements, lower, upper, p, elements, p.oldSize, freeList, edges + size_t(c) * 5, degree + numVars, numTasks));
				for (uint j = lower; j < upper; j++) {
					c += degree[p.firstVar + j + numVars];
					freeList += degree[p.firstVar + j + numVars * 4] * ELEMENT_WIDTH;
				}
			}
		}
	}
	if (p.tmpSize != 0) {
		memset(elements + p.oldSize + p.deltaSize, -1, headSize * sizeof(uint));
		freeList += headSize;
		if (numTmpEdges != 0) {
			for (uint i = 0; i < nSegs; i++) {
				lower = i * segSize;
				upper = (i == nSegs - 1) ? numPartialVars : lower + segSize;
				ioServ.post(boost::bind(edges2Elements, lower, upper, p, elements, p.oldSize + p.deltaSize, freeList, edges + size_t(c) * 5, degree + numVars * 2, numTasks));
				for (uint j = lower; j < upper; j++) {
					c += degree[p.firstVar + j + numVars * 2];
					freeList += degree[p.firstVar + j + numVars * 5] * ELEMENT_WIDTH;
				}
			}
		}
	}
	unique_lock<mutex> lck(comp_mtx);
	while (!finished) cv.wait(lck);
	delete[] edges;
	return elements;
}

uint* readElements_n(Partition p, int k)
{
	string str = "Partition" + to_string(k);
	FILE *fp = fopen(str.c_str(), "rb");
	uint poolSize = p.oldSize + p.deltaSize + p.tmpSize;
	uint *elements = new uint[poolSize];
	fread(elements, sizeof(uint), poolSize, fp);
	fclose(fp);
	return elements;
}

uint* readElements(Partition p, int k, uint *degree, boost::asio::io_service &ioServ)
{
	if (COMPRESSED) {
		return readElements_c(p, k, degree, ioServ);
	} else {
		return readElements_n(p, k);
	}
}

void readAndTransferElements(Partition p, int k, int start, uint heapSize, uint *elementPool, uint *degree, boost::asio::io_service &ioServ)
{
	cout << "Reading Elements..." << flush;
	uint *elements = readElements(p, k, degree, ioServ);
	cout << "OK." << endl;
	cout << "Transferring Elements..." << flush;
	transferElements(p, elements, start, heapSize, elementPool);
	cout << "OK." << endl;
}

uint* transferBackElements(Partition p, int start, uint heapSize, uint *elementPool)
{
	uint poolSize = p.oldSize + p.deltaSize + p.tmpSize;
	uint *elements = new uint[poolSize];
	cudaSafeCall(cudaMemcpy(elements, elementPool + (heapSize / 2) * start, poolSize * sizeof(uint), D2H));
	return elements;
}

/*
int popcnt(uint num)
{
int r = 0;
while (num)
{
num &= num - 1;
r++;
}
return r;
}
*/

int ffs(uint num)
{
	if (num == 0) return 0;
	uint t = 1;
	int r = 1;
	while ((num & t) == 0) {
		t = t << 1;
		r++;
	}
	return r;
}

void elements2Edges(uint lower, uint upper, uint virtualNumPartialVars, uint *elements, uint offset, uchar *edges, uint numTasks)
{
	for (uint i = lower; i < upper; i++) {
		for (int j = 0; j < numLabels; j++) {
			uint headIndex = offset + virtualNumPartialVars * j + i;
			uint index = elements[headIndex];
			while (index != NIL) {
				uint base = elements[index + BASE];
				for (int k = 0; k < BASE; k++) {
					uint myBits = elements[index + k];
					while (myBits) {
						int pos = ffs(myBits) - 1;
						myBits &= (myBits - 1);
						uint dst = base * ELEMENT_CARDINALITY + mul32(k) + pos;
						*((uint*)edges) = dst;
						*(edges + 4) = j + 1;
						edges += 5;
					}
				}
				index = elements[index + NEXT];
			}
		}
	}
	unique_lock<mutex> lck(addEdges_mtx);
	numFinished++;
	if (numFinished == numTasks) {
		finished = true;
		cv.notify_one();
	}
}

void storeElements_c(Partition p, int k, uint *elements, uint *degree, boost::asio::io_service &ioServ)
{
	uint numOldEdges = 0, numDeltaEdges = 0, numTmpEdges = 0;
	for (uint i = p.firstVar; i <= p.lastVar; i++) {
		numOldEdges += degree[i];
		numDeltaEdges += degree[i + numVars];
		numTmpEdges += degree[i + numVars * 2];
	}
	uint numEdges = numOldEdges + numDeltaEdges + numTmpEdges;
	size_t size = size_t(numEdges) * 5;
	uchar *edges = new uchar[size];
	uint numPartialVars = p.lastVar - p.firstVar + 1;
	uint virtualNumPartialVars = roundToNextMultipleOf(numPartialVars);
	uint segSize = numPartialVars / 16 + 1;
	uint nSegs = numPartialVars / segSize;
	if (numPartialVars % segSize != 0) nSegs++;
	uint numTasks = 0;
	if (numOldEdges != 0) numTasks += nSegs;
	if (numDeltaEdges != 0) numTasks += nSegs;
	if (numTmpEdges != 0) numTasks += nSegs;
	numFinished = 0;
	finished = false;
	uint c = 0;
	uint lower, upper;
	if (numOldEdges != 0) {
		for (uint i = 0; i < nSegs; i++) {
			lower = i * segSize;
			upper = (i == nSegs - 1) ? numPartialVars : lower + segSize;
			ioServ.post(boost::bind(elements2Edges, lower, upper, virtualNumPartialVars, elements, 0, edges + size_t(c) * 5, numTasks));
			for (uint j = lower; j < upper; j++)
				c += degree[p.firstVar + j];
		}
	}
	if (numDeltaEdges != 0) {
		for (uint i = 0; i < nSegs; i++) {
			lower = i * segSize;
			upper = (i == nSegs - 1) ? numPartialVars : lower + segSize;
			ioServ.post(boost::bind(elements2Edges, lower, upper, virtualNumPartialVars, elements, p.oldSize, edges + size_t(c) * 5, numTasks));
			for (uint j = lower; j < upper; j++)
				c += degree[p.firstVar + j + numVars];
		}
	}
	if (numTmpEdges != 0) {
		for (uint i = 0; i < nSegs; i++) {
			lower = i * segSize;
			upper = (i == nSegs - 1) ? numPartialVars : lower + segSize;
			ioServ.post(boost::bind(elements2Edges, lower, upper, virtualNumPartialVars, elements, p.oldSize + p.deltaSize, edges + size_t(c) * 5, numTasks));
			for (uint j = lower; j < upper; j++)
				c += degree[p.firstVar + j + numVars * 2];
		}
	}
	unique_lock<mutex> lck(comp_mtx);
	while (!finished) cv.wait(lck);
	delete[] elements;
	string str = "Partition" + to_string(k);
	FILE *fp = fopen(str.c_str(), "wb+");
	fwrite(edges, 1, size, fp);
	fclose(fp);
	delete[] edges;
}

void storeElements_n(Partition p, int k, uint *elements)
{
	string str = "Partition" + to_string(k);
	FILE *fp = fopen(str.c_str(), "wb+");
	uint poolSize = p.oldSize + p.deltaSize + p.tmpSize;
	fwrite(elements, sizeof(uint), poolSize, fp);
	fclose(fp);
	delete[] elements;
}

void storeElements(Partition p, int k, uint *elements, uint *degree, boost::asio::io_service &ioServ)
{
	if (COMPRESSED) {
		// stored in CSR format
		storeElements_c(p, k, elements, degree, ioServ);
	} else {
		storeElements_n(p, k, elements);
	}
}

void transferBackAndStoreElements(Partition p, int k, int start, uint heapSize, uint *elementPool, uint *degree, boost::asio::io_service &ioServ)
{
	getDegree(p, start, degree);
	cout << "Transferring Back Elements..." << flush;
	uint *elements = transferBackElements(p, start, heapSize, elementPool);
	cout << "OK." << endl;
	cout << "Storing Elements..." << flush;
	storeElements(p, k, elements, degree, ioServ);
	cout << "OK." << endl;
}

void repartRegion(uint *fromElements, uint fromOffset, uint *toElements1, uint *toElements2, uint &k1, uint &k2, uint numPartialVars1, uint numPartialVars2)
{
	uint virtualNumPartialVars1 = roundToNextMultipleOf(numPartialVars1);
	uint virtualNumPartialVars2 = roundToNextMultipleOf(numPartialVars2);
	uint headSize1 = virtualNumPartialVars1 * numLabels;
	uint headSize2 = virtualNumPartialVars2 * numLabels;
	uint virtualNumPartialVars = roundToNextMultipleOf(numPartialVars1 + numPartialVars2);
	uint toOffset1 = k1;
	uint toOffset2 = k2;
	memset(toElements1 + toOffset1, -1, headSize1 * sizeof(uint));
	k1 += headSize1;
	memset(toElements2 + toOffset2, -1, headSize2 * sizeof(uint));
	k2 += headSize2;
	for (int i = 0; i < numLabels; i++) {
		for (uint j = 0; j < numPartialVars1; j++) {
			uint fromHeadIndex = fromOffset + virtualNumPartialVars * i + j;
			uint fromIndex = fromElements[fromHeadIndex];
			if (fromIndex != NIL) {
				uint toHeadIndex1 = toOffset1 + virtualNumPartialVars1 * i + j;
				toElements1[toHeadIndex1] = k1;
				uint toIndex1 = k1;
				k1 += ELEMENT_WIDTH;
				for (;;) {
					memcpy(toElements1 + toIndex1, fromElements + fromIndex, ELEMENT_WIDTH * sizeof(uint));
					fromIndex = fromElements[fromIndex + NEXT];
					if (fromIndex == NIL) break;
					toElements1[toIndex1 + NEXT] = k1;
					toIndex1 = k1;
					k1 += ELEMENT_WIDTH;
				}
			}
		}
		for (uint j = 0; j < numPartialVars2; j++) {
			uint fromHeadIndex = fromOffset + virtualNumPartialVars * i + numPartialVars1 + j;
			uint fromIndex = fromElements[fromHeadIndex];
			if (fromIndex != NIL) {
				uint toHeadIndex2 = toOffset2 + virtualNumPartialVars2 * i + j;
				toElements2[toHeadIndex2] = k2;
				uint toIndex2 = k2;
				k2 += ELEMENT_WIDTH;
				for (;;) {
					memcpy(toElements2 + toIndex2, fromElements + fromIndex, ELEMENT_WIDTH * sizeof(uint));
					fromIndex = fromElements[fromIndex + NEXT];
					if (fromIndex == NIL) break;
					toElements2[toIndex2 + NEXT] = k2;
					toIndex2 = k2;
					k2 += ELEMENT_WIDTH;
				}
			}
		}
	}
}

int getNumRegions(Partition p)
{
	int n = 0;
	if (p.oldSize != 0) n++;
	if (p.deltaSize != 0) n++;
	if (p.tmpSize != 0) n++;
	return n;
}

void repartElements(vector<Partition> &partitions, int s, int d, uint *elements, uint* &elements1, uint* &elements2, uint *degree)
{
	cout << "Repartitioning..." << flush;
	uint headSize = roundToNextMultipleOf(partitions[s].lastVar - partitions[s].firstVar + 1) * numLabels;
	int numRegions = getNumRegions(partitions[s]);
	uint numElements = (partitions[s].oldSize + partitions[s].deltaSize + partitions[s].tmpSize - headSize * numRegions) / ELEMENT_WIDTH;
	uint tmp1 = 0;
	uint tmp2 = 0;
	uint tmp3 = partitions[s].lastVar;
	uint numElements1, numElements2;
	for (uint i = partitions[s].firstVar; i <= partitions[s].lastVar; i++) {
		uint numElements_i = degree[i + numVars * 3] + degree[i + numVars * 4] + degree[i + numVars * 5];
		tmp1 += numElements_i;
		if (tmp1 > numElements / 2) {
			partitions[s].lastVar = i - 1;
			numElements1 = tmp1 - numElements_i;
			numElements2 = numElements - numElements1;
			tmp2 = i;
			break;
		}
	}
	partitions.insert(partitions.begin() + d, Partition(tmp2, tmp3, 0, 0, 0, true));
	uint deltaOffset = partitions[s].oldSize;
	uint tmpOffset = partitions[s].oldSize + partitions[s].deltaSize;
	uint numPartialVars1 = partitions[s].lastVar - partitions[s].firstVar + 1;
	uint numPartialVars2 = partitions[d].lastVar - partitions[d].firstVar + 1;
	elements1 = new uint[numElements1 * ELEMENT_WIDTH + roundToNextMultipleOf(numPartialVars1) * numLabels * numRegions];
	elements2 = new uint[numElements2 * ELEMENT_WIDTH + roundToNextMultipleOf(numPartialVars2) * numLabels * numRegions];
	uint k1 = 0, k2 = 0;
	if (partitions[s].oldSize != 0) {
		repartRegion(elements, 0, elements1, elements2, k1, k2, numPartialVars1, numPartialVars2);
		partitions[s].oldSize = k1;
		partitions[d].oldSize = k2;
	}
	if (partitions[s].deltaSize != 0) {
		repartRegion(elements, deltaOffset, elements1, elements2, k1, k2, numPartialVars1, numPartialVars2);
		partitions[s].deltaSize = k1 - partitions[s].oldSize;
		partitions[d].deltaSize = k2 - partitions[d].oldSize;
	}
	if (partitions[s].tmpSize != 0) {
		repartRegion(elements, tmpOffset, elements1, elements2, k1, k2, numPartialVars1, numPartialVars2);
		partitions[s].tmpSize = k1 - partitions[s].oldSize - partitions[s].deltaSize;
		partitions[d].tmpSize = k2 - partitions[d].oldSize - partitions[d].deltaSize;
	}
	delete[] elements;
	cout << "OK." << endl;
}

void renamePartitions(int numPartitions, int k)
{
	for (int i = numPartitions - 1; i >= k; i--) {
		string oldName, newName;
		oldName = "Partition" + to_string(i);
		newName = "Partition" + to_string(i + 1);
		rename(oldName.c_str(), newName.c_str());
	}
}

void repartition(vector<Partition> &partitions, int s, int d, int start, uint heapSize, uint *elementPool, uint *degree, boost::asio::io_service &ioServ)
{
	getDegree(partitions[s], start, degree);
	uint *elements = transferBackElements(partitions[s], start, heapSize, elementPool);
	uint *elements1, *elements2;
	repartElements(partitions, s, d, elements, elements1, elements2, degree);
	transferElements(partitions[s], elements1, start, heapSize, elementPool);
	renamePartitions(partitions.size(), d);
	storeElements(partitions[d], d, elements2, degree, ioServ);
}

void precomputation(char *filePath, vector<Partition> &partitions, map<string, int> labelMap, uint *degree, uint heapSize, uint *elementPool, boost::asio::io_service &ioServ)
{
	cout << "##### SUPERSTEP 0 #####" << endl;
	for (int i = 0; i < partitions.size(); i++) {
		cout << "== PARTITION " << i << " ==" << endl;
		// load Partition p
		if (partitions[i].flag == false) {
			uint numKeys, *valIndex;
			uint *edges = readEdges(filePath, labelMap, partitions[i], numKeys, valIndex);
			uint *elements = createGraph(partitions[i], edges, numKeys, valIndex, degree, ioServ);
			transferElements(partitions[i], elements, 0, heapSize, elementPool);
		} else {
			readAndTransferElements(partitions[i], i, 0, heapSize, elementPool, degree, ioServ);
		}
		// self-matching
		int iterNo = 0;
		for (;;) {
			cout << "## ITERATION " << ++iterNo << " ##" << endl;
			bool r = false;
			do {
				if (r) {
					repartition(partitions, i, i + 1, 0, heapSize, elementPool, degree, ioServ);
					r = false;
				}
				spagpu_s(partitions[i], 0, heapSize, elementPool, r);
			} while (r);
			mergeAndDiff(partitions[i], heapSize, elementPool);
			if (partitions[i].deltaSize == 0) {
				partitions[i].deltaSize = partitions[i].oldSize;
				partitions[i].oldSize = 0;
				transferBackAndStoreElements(partitions[i], i, 0, heapSize, elementPool, degree, ioServ);
				break;
			}
		}
	}
}

void run_computation(vector<Partition> &partitions, uint heapSize, uint *elementPool, uint *degree, boost::asio::io_service &ioServ)
{
	for (int i = 0; i < partitions.size(); i++) {
		bool r1 = false;
		bool r2 = false;
		// load Partition p1
		readAndTransferElements(partitions[i], i, 0, heapSize, elementPool, degree, ioServ);

		for (int j = i + 1; j < partitions.size(); j++) {
			// load Partition p2
			readAndTransferElements(partitions[j], j, 1, heapSize, elementPool, degree, ioServ);

			// rule application
			std::cout << "== COMP START ==" << std::endl;
			do {
				if (r1) {
					repartition(partitions, i, j + 1, 0, heapSize, elementPool, degree, ioServ);
					r1 = false;
				}
				if (r2) {
					repartition(partitions, j, j + 1, 1, heapSize, elementPool, degree, ioServ);
					r2 = false;
				}
				spagpu_b(partitions[i], partitions[j], r1, r2, heapSize, elementPool);
			} while (r1 || r2);

			do {
				if (r1) {
					repartition(partitions, i, j + 1, 0, heapSize, elementPool, degree, ioServ);
					r1 = false;
				}
				if (r2) {
					repartition(partitions, j, j + 1, 1, heapSize, elementPool, degree, ioServ);
					r2 = false;
				}
				spagpu(partitions[i], partitions[j], r1, r2, heapSize);
			} while (r1 || r2);
			std::cout << "== COMP END ==" << std::endl;

			transferBackAndStoreElements(partitions[j], j, 1, heapSize, elementPool, degree, ioServ);
		}
		// update
		mergeAndDiff(partitions[i], heapSize, elementPool);

		transferBackAndStoreElements(partitions[i], i, 0, heapSize, elementPool, degree, ioServ);
	}
}

int main(int argc, char** argv)
{
	clock_t startTime = clock();

	// PREPROCESSING
	cout << "===== PREPROCESSING INFO =====" << endl;
	checkGPUConfiguration();

	map<string, int> labelMap;
	readAndTransferRules(argv[1], labelMap);

	uint *degree = getGraphInfo(argv[2], labelMap);

	// amount of memory reserved for the graph
	// in 32-bit words
	uint heapSize, *elementPool;
	allocateElementPool(heapSize, elementPool);

	// divide vertices into logical intervals
	vector<Partition> partitions;
	initializePartitions(partitions, degree, heapSize);

	// COMPUTATION
	boost::asio::io_service ioServ;
	boost::thread_group threadPool;
	boost::asio::io_service::work work(ioServ);
	for (int i = 0; i < NUMTHREADS; i++)
		threadPool.create_thread(boost::bind(&boost::asio::io_service::run, &ioServ));

	int superstepNo = 0;
	precomputation(argv[2], partitions, labelMap, degree, heapSize, elementPool, ioServ);

	for (;;) {
		cout << "##### SUPERSTEP " << ++superstepNo << " #####" << endl;
		run_computation(partitions, heapSize, elementPool, degree, ioServ);

		// The computation is finished when no new edges can be added.
		bool done = true;
		for (int i = 0; i < partitions.size(); i++) {
			if (partitions[i].deltaSize != 0)
				done = false;
		}
		if (done) {
			cudaFree(elementPool);
			break;
		}
	}

	// REPORTING RESULTS
	uint numEdges = 0;
	for (uint i = 0; i < numVars; i++)
		numEdges += degree[i];
	delete[] degree;

	cout << "===== SPAGPU FINISHED =====" << endl;
	cout << "NUM SUPERSTEPS: " << superstepNo << endl;
	cout << "FINAL NUM PARTITIONS: " << partitions.size() << endl;
	cout << "FINAL NUM EDGES: " << numEdges << endl;
	cout << "TOTAL SPAGPU TIME: " << getElapsedTime(startTime) << "s." << endl;

	return 0;
}


