#pragma once
/******************************************************************************
Date:  2022/09
Author: CHU-MIN, NIEN
Description:
******************************************************************************/

#include <math.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstdlib>
#include <chrono>//random seed
#include <random> // normal_distribution random
#include <cmath>
#include <stdio.h>
#include "opencv2/opencv.hpp"
#include "CNNDataset.h"
#include "CPUCNNLayer.h"

using namespace std;
using namespace cv;

// CNNCls
typedef vector<CPUCNNLayer> VECCPUCNNLayers;

//Builder some VECCPUCNNLayers that you want
class CPUCNNLayerCreater
{
public:
	VECCPUCNNLayers vec_layers_;

	CPUCNNLayerCreater() {};
	CPUCNNLayerCreater(CPUCNNLayer layer) {
		vec_layers_.push_back(layer);
	}
	void AddLayer(CPUCNNLayer layer)
	{
		vec_layers_.push_back(layer);
	}
};

class CPUCNN
{
private:
	VECCPUCNNLayers vec_layers_;
	int layer_num_;
	int batch_size_;
	float eta_conv_;
	float alpha_conv_;
	float eta_fc_;
	float alpha_fc_;

public:
	CPUCNN(CPUCNNLayerCreater layer_creater, int batch_size)
	{
		eta_conv_ = 0.0; //learning rate: ReLU using 0.0003, Sigmoid using 0.01 
		alpha_conv_ = 0.0;//momentum rate: 0.1 
		eta_fc_ = 0.00003; //learning rate: ReLU using 0.0003, Sigmoid using 0.01 
		alpha_fc_ = 0.01;//momentum rate: 0.1 
		vec_layers_ = layer_creater.vec_layers_;
		layer_num_ = vec_layers_.size();
		this->batch_size_ = batch_size;
		SetupTest(batch_size);

	};
	~CPUCNN() {};
	void SetBatchsize(int batchsize) {
		this->batch_size_ = batchsize;
	}
	void Train(float**** train_x, float** train_label, int numofimage);
	void Inference(float**** test_x, float** test_label, int numOfImage);
	void Setup(int batch_size);// builder CPUCNN with batch_size_ and initialize some parameters of each layer
	void SetupTest(int batch_size);

	//back-propagation
	void BackPropagation(float**** x, float** y);
	void SetOutLayerErrors(float**** input_maps, float** target_labels);
	void SetHiddenLayerErrors();
	void SetFCHiddenLayerErrors(CPUCNNLayer& Lastlayer, CPUCNNLayer& layer, CPUCNNLayer& nextLayer);
	void SetSampErrors(CPUCNNLayer& layer, CPUCNNLayer& nextLayer);
	void SetConvErrors(CPUCNNLayer& layer, CPUCNNLayer& nextLayer);

	void UpdateKernels(CPUCNNLayer& layer, CPUCNNLayer& lastLayer, char* str_File_Kernel, float eta, float alpha);
	void UpdateBias(CPUCNNLayer& layer, char* str_File_Bias, float eta);
	void UpdateParas();

	//forward
	void Forward(float**** x);
	void SetInLayerOutput(float**** x);
	void SetConvOutput(CPUCNNLayer& layer, CPUCNNLayer& lastLayer);
	void SetSampOutput(CPUCNNLayer& layer, CPUCNNLayer& lastLayer);
	void SetFCHLayerOutput(CPUCNNLayer& layer, CPUCNNLayer& lastLayer);
	void SetOutLayerOutput(CPUCNNLayer& layer, CPUCNNLayer& lastLayer);

	//load parameter
	void LoadParas();
	void LoadBias(CPUCNNLayer& layer, char* str_File_Bias);
	void LoadKernels(CPUCNNLayer& layer, CPUCNNLayer& lastLayer, char* str_File_Kernel);
};

