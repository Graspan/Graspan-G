#include "ddm.h"

std::string getItem(std::istringstream &lineStream){
    std::string item;
    getline(lineStream, item, '\t');
    return item;
}

void DDM::initialize(char *filename, std::vector<Partition> &partitions, uint *degree){
	ddm.assign(partitions.size(), std::vector<long>(partitions.size(), 0));
	std::ifstream fin;
	fin.open(filename);
	std::string line;
	getline(fin, line);//skip first comment
    getline(fin, line);//second comment
    std::istringstream lineStream(line);
    uint numEdges = std::atoi(getItem(lineStream).c_str());
	getline(fin, line);
	for (uint i = 0; i < numEdges; i++) {
		getline(fin, line);
		std::istringstream lineStream(line);
		std::string item;
		getline(lineStream, item, '\t');
		uint src = std::stoi(item);
		getline(lineStream, item, '\t');
		uint dst = std::stoi(item);
		int fromPid, toPid;
		for (int j = 0; j < partitions.size(); j++) {
			if (src >= partitions[j].firstVar && src <= partitions[j].lastVar)
				fromPid = j;
			if (dst >= partitions[j].firstVar && dst <= partitions[j].lastVar)
				toPid = j;
		}
		if (fromPid != toPid)
			ddm[fromPid][toPid] += degree[dst];
	}
	fin.close();

	ddm_t.assign(partitions.size(), std::vector<bool>(partitions.size(), false));
	for (int i = 0; i < ddm.size() - 1; i++) {
		for (int j = i + 1; j < ddm.size(); j++) {
				ddm_t[i][j] = true;
		}
	}
	flag = false;
}

bool DDM::nextPair(int &p, int &q){
	bool res = false;
	long max = -1;
LY:	for (int i = 0; i < ddm.size() - 1; i++) {
		for (int j = i + 1; j < ddm.size(); j++) {
				if (ddm[i][j] + ddm[j][i] > max) {
                    res = true;
					max = ddm[i][j] + ddm[j][i];
					p = i;
					q = j;
				}
		}
	}
    /*
	if (!res) {
		if (f) {
			for (int i = 0; i < ddm.size() - 1; i++)
				for (int j = i + 1; j < ddm.size(); j++)
					ddm_t[i][j] = true;
			f = false;
			res = false; max = -1; goto LY;
		}
	}
    */
	return res;
}

void DDM::update(int p, int q, int rs){
	if (rs == 0) {
		for (int i = 0; i < ddm.size(); i++) {
			ddm[i].push_back(ddm[i][p] / 2);
		}
		ddm[p][ddm.size()] = 1;
		ddm.push_back(std::vector<long>(ddm[0].size(), 0));
		for (int i = 0; i < ddm[0].size() - 1; i++) {
			ddm[ddm.size() - 1][i] = ddm[p][i] / 2;
		}
		ddm[ddm.size() - 1][p] = 1;
		for (int i = 0; i < ddm.size() - 1; i++) {
			ddm[p][i] = ddm[p][i] / 2;
			ddm[i][p] = ddm[i][p] / 2;
		}
		for (int i = 0; i < p; i++) {
			ddm_t[i].push_back(ddm_t[i][p]);
		}
		ddm_t[p].push_back(true);
		for (int i = p + 1; i < ddm_t.size(); i++) {
			ddm_t[i].push_back(ddm_t[p][i]);
		}
		ddm_t.push_back(std::vector<bool>(ddm_t[0].size(), false));
	}
	if (rs == 1) {
		for (int i = 0; i < ddm.size(); i++) {
			ddm[i].push_back(ddm[i][q] / 2);
		}
		ddm[q][ddm.size()] = 1;
		ddm.push_back(std::vector<long>(ddm[0].size(), 0));
		for (int i = 0; i < ddm[0].size() - 1; i++) {
			ddm[ddm.size() - 1][i] = ddm[q][i] / 2;
		}
		ddm[ddm.size() - 1][q] = 1;
		for (int i = 0; i < ddm.size() - 1; i++) {
			ddm[q][i] = ddm[q][i] / 2;
			ddm[i][q] = ddm[i][q] / 2;
		}
		for (int i = 0; i < q; i++) {
			ddm_t[i].push_back(ddm_t[i][q]);
		}
		ddm_t[q].push_back(true);
		for (int i = q + 1; i < ddm_t.size(); i++) {
			ddm_t[i].push_back(ddm_t[q][i]);
		}
		ddm_t.push_back(std::vector<bool>(ddm_t[0].size(), false));
	}
	if (rs == -1) {
		if (p > q)
			ddm_t[q][p] = false;
		else
			ddm_t[p][q] = false;
	}
    /*
	if (numNewSuperstepsEdges > 0)
		flag = true;
    */
}


















