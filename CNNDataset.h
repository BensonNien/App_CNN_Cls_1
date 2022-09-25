#pragma once
#include <fstream> 
#include "opencv2/opencv.hpp"
#include "CNNDataStruct.h"

#define IMG_H 64 //image's height
#define IMG_W 32 //image's width


using namespace std;
using namespace cv;

class CNNDataset {
public:
	// Big5 Funcs
	CNNDataset() {};
	~CNNDataset() {};

	// Static Member Funcs
	static void Load(double**** train_x, double**** test_x, double** train_label, double** test_label);
};
