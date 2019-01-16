NVCC=nvcc
NVCCFLAGS=-arch=sm_61
CXX=g++
CXXFLAGS=-std=c++11 -I/usr/local/cuda-9.2/include
LDLIBS=-L/usr/local/cuda-9.2/lib64 -lcudart -lpthread -lboost_thread -lboost_system

comp: kernel.o
	$(CXX) $(CXXFLAGS) -o ../bin/comp main.cpp kernel.o $(LDLIBS)

kernel.o:
	$(NVCC) $(NVCCFLAGS) -c kernel.cu

.PHONY: clean
clean:
	rm *.o ../bin/comp