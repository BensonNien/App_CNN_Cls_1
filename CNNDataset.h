#pragma once
#include <fstream> 
#include "opencv2/opencv.hpp"
#include "CNNDataStruct.h"

using namespace std;
using namespace cv;

class CNNDataset {
public:
	// Big5 Funcs
	CNNDataset() {};
	~CNNDataset() {};

	// Static Member Funcs
	static void Load(DatasetLoadingParamPKG& r_dataset_param);
};
