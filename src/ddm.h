#ifndef DDM_H
#define DDM_H

#include "kernel.h"

class DDM{

private:
    std::vector<std::vector<long>> ddm;
    std::vector<std::vector<bool>> ddm_t;
    bool flag;

public:
    DDM(){}
    void initialize(char *filename, std::vector<Partition> &partitions, uint *degree);
    bool nextPair(int &p, int &q);
    void update(int p, int q, int rs);

};

#endif