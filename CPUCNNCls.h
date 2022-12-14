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
	size_t layer_num_;
	size_t batch_size_;
	float eta_conv_;
	float alpha_conv_;
	float eta_fc_;
	float alpha_fc_;

public:
	CPUCNN(CPUCNNLayerCreater layer_creater, int batch_size)
	{
		eta_conv_ = 0.006; //learning rate 
		alpha_conv_ = 0.2;//momentum rate
		eta_fc_ = 0.006; //learning rate
		alpha_fc_ = 0.2;//momentum rate
		vec_layers_ = layer_creater.vec_layers_;
		layer_num_ = vec_layers_.size();
		this->batch_size_ = batch_size;
		Setup(batch_size);
		SetupTest(batch_size);

	};
	~CPUCNN() {};
	void SetBatchsize(int batchsize) {
		this->batch_size_ = batchsize;
	}
	void Train(DatasetLoadingParamPKG& r_dataset_param);
	void Inference(DatasetLoadingParamPKG& r_dataset_param);
	void Setup(int batch_size);// builder CPUCNN with batch_size_ and initialize some parameters of each layer
	void SetupTest(int batch_size);

	//back-propagation
	void BackPropagation(float* p_batch_data, float* p_batch_label);
	void SetOutLayerErrors(float* p_input_maps, float* p_target_labels);
	void SetHiddenLayerErrors();
	void SetFCHiddenLayerErrors(CPUCNNLayer& Lastlayer, CPUCNNLayer& layer, CPUCNNLayer& nextLayer);
	void SetSampErrors(CPUCNNLayer& layer, CPUCNNLayer& nextLayer);
	void SetConvErrors(CPUCNNLayer& layer, CPUCNNLayer& nextLayer);

	void UpdateKernels(CPUCNNLayer& layer, CPUCNNLayer& lastLayer, char* str_File_Kernel, float eta, float alpha);
	void UpdateBias(CPUCNNLayer& layer, char* str_File_Bias, float eta);
	void UpdateParas();

	//forward
	void Forward(float* p_batch_data);
	void SetInLayerOutput(float* p_batch_data);
	void SetConvOutput(CPUCNNLayer& layer, CPUCNNLayer& lastLayer);
	void SetSampOutput(CPUCNNLayer& layer, CPUCNNLayer& lastLayer);
	void SetFCHLayerOutput(CPUCNNLayer& layer, CPUCNNLayer& lastLayer);
	void SetOutLayerOutput(CPUCNNLayer& layer, CPUCNNLayer& lastLayer);

	//load parameter
	void LoadParas();
	void LoadBias(CPUCNNLayer& layer, char* str_File_Bias);
	void LoadKernels(CPUCNNLayer& layer, CPUCNNLayer& lastLayer, char* str_File_Kernel);
};

