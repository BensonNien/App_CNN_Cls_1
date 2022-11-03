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

#define RELU(x) x //the relu active function
#define SIGMOID(x) (1/(1+exp(-x))) //the sigmoid active function

using namespace std;
using namespace cv;
// Utility
double cpu_time();
void randomMatrix(int x, int y, double** outmatrix);
void convnValid(double** matrix, double** kernel, int m, int n, int km, int kn, double** outmatrix);// m n is the dimension of matrix and km kn is the dimension of kernel, outmatrix is result
void Relu(double** matrix, double bias, int m, int n);// m n is the dimension of matrix
void Sigmoid(double** matrix, double bias, int m, int n);// m n is the dimension of matrix
void Expone(double** matrix, double bias, int m, int n);// m n is the dimension of matrix
void ArrayPlus(double** x, double** y, int m, int n);
void scaleMatrix(double** lastMap, RectSize scaleSize, int m, int n, double** sampling);//sampling
void rot180(double** matrix, int m, int n, double** rotMatrix);
void convnFull(double** matrix, double** kernel, int m, int n, int km, int kn, double** outmatrix, double** extendMatrix);// convn full mode
void matrixDrelu(double** matrix, int m, int n, double** M);// calculate derivation of ReLU function with matrix
void matrixDreluFChidden(double** matrix, int m, int n, double* M);// calculate derivation of ReLU function in FChiddenlayer with matrix
void matrixDsigmoid(double** matrix, int m, int n, double** M);// calculate derivation of sigmoid function with matrix
void matrixDsigmoidFChidden(double** matrix, int m, int n, double* M);// calculate derivation of sigmoid function in FChiddenlayer with matrix
void kronecker(double** matrix, RectSize scale, int m, int n, double** outmatrix);
void matrixMultiply(double** matrix1, double** matrix2, int m, int n);//inner product of matrix 1 and matrix 2, result is matrix1
void sum(double**** errors, int j, int m, int n, int batchSize, double** outmatrix);
double sum(double** error, int m, int n);
void ArrayDivide(double** matrix, int batchSize, int m, int n);// result is matrix;
void ArrayMultiply(double** matrix, double val, int m, int n);// array multiply a double value, result in matrix
void setValue(double** maps, double** sum, int m, int n);
int findIndex(double*** p);
int findIndex(double* p);

// Layer

class Layer
{
private:
	int outMapNum;
	char type;
	RectSize mapSize;
	RectSize scaleSize;
	RectSize kernelSize;
	int classNum;

public:
	Layer() {};
	~Layer() {
		// 		delete []kernel;
		// 		delete []outputmaps;
		// 		delete []errors;
		// 		delete []bias;
	};
	double**** kernel;
	double**** laststepdeltakernel;//for adding momentum
	double**** outputmaps;
	double**** errors;
	double* bias;

	Layer buildInputLayer(int InputLayerOutMapNum, RectSize mapSize);
	Layer buildConvLayer(int outMapNum, RectSize kernelize);
	Layer buildSampLayer(RectSize scaleSize);
	Layer buildFullyConnectedHiddenLayer(int classNum);
	Layer buildOutputLayer(int classNum);

	void initKernel(int frontMapNum);
	void initLastStepDeltaKernel(int frontMapNum);//for adding momentum
	void initOutputKernel(int frontMapNum, RectSize s);
	void initOutputLastStepDeltaKernel(int frontMapNum, RectSize s);//for adding momentum
	void initErros(int batchSize);
	void initOutputmaps(int batchSize) {
		outputmaps = new double*** [batchSize];
		for (int i = 0; i < batchSize; i++)
		{
			outputmaps[i] = new double** [outMapNum];
			for (int j = 0; j < outMapNum; j++)
			{
				outputmaps[i][j] = new double* [getMapSize().x];
				for (int ii = 0; ii < getMapSize().x; ii++)
				{
					outputmaps[i][j][ii] = new double[getMapSize().y];
				}
			}
		}
	}
	void initBias(int frontmapNum, int no_iter) {

		bias = new double[outMapNum];
		if (no_iter % 3 == 0) {
			for (int i = 0; i < outMapNum; i++)
			{
				//bias[i]=0.0;
				bias[i] = 0.1;
			}
		}
		else {
			for (int i = 0; i < outMapNum; i++)
			{
				//bias[i]=1.0;
				bias[i] = 0.1;
			}
		}
	}

	void setError(int numBatch, int mapNo, int mapX, int mapY, double v);
	double** getError(int numBatch, int mapNo) {
		return errors[numBatch][mapNo];
	}
	void setError(int numBatch, int mapNo, double** matrix, int m, int n);

	double** getKernel(int numBatch, int mapNo) {
		return kernel[numBatch][mapNo];
	}

	int getOutMapNum() {
		return outMapNum;
	}
	char getType() {
		return type;
	}
	RectSize getMapSize() {
		return mapSize;
	}
	void setMapSize(RectSize mapSize) {
		this->mapSize = mapSize;
	}
	void setOutMapNum(int outMapNum) {
		this->outMapNum = outMapNum;
	}
	RectSize getKernelSize() {
		return kernelSize;
	}
	RectSize getScaleSize() {
		return scaleSize;
	}

};


// CNNCls
typedef vector<Layer> layers;

//Builder some layers that you want
class LayerBuilder
{
public:
	layers mLayers;

	LayerBuilder() {};
	LayerBuilder(Layer layer) {
		mLayers.push_back(layer);
	}
	void addLayer(Layer layer)
	{
		mLayers.push_back(layer);
	}
};

class CNN
{
private:
	layers m_layers;
	int layerNum;
	int batchSize;
	double ETA_CONV;
	double ALPHA_CONV;
	double ETA_FC;
	double ALPHA_FC;

public:
	CNN(LayerBuilder layerBuilder, int batchSize)
	{
		ETA_CONV = 0.0; //learning rate: ReLU using 0.0003, Sigmoid using 0.01 
		ALPHA_CONV = 0.0;//momentum rate: 0.1 
		ETA_FC = 0.00003; //learning rate: ReLU using 0.0003, Sigmoid using 0.01 
		ALPHA_FC = 0.01;//momentum rate: 0.1 
		m_layers = layerBuilder.mLayers;
		layerNum = m_layers.size();
		this->batchSize = batchSize;
		setup(batchSize);

	};
	~CNN() {};
	void setBatchsize(int batchsize) {
		this->batchSize = batchsize;
	}
	void train(double**** train_x, double** train_label, int numofimage);
	void test(double**** test_x, double** test_label, int numOfImage);
	void setup(int batchSize);// builder CNN with batchSize and initialize some parameters of each layer

	//back-propagation
	void backPropagation(double**** x, double** y);
	void setOutLayerErrors(double**** x, double** y);
	void setHiddenLayerErrors();
	void setFCHiddenLayerErrors(Layer& Lastlayer, Layer& layer, Layer& nextLayer);
	void setSampErrors(Layer& layer, Layer& nextLayer);
	void setConvErrors(Layer& layer, Layer& nextLayer);

	void updateKernels(Layer& layer, Layer& lastLayer, char* str_File_Kernel, double eta, double alpha);
	void updateBias(Layer& layer, char* str_File_Bias, double eta);
	void updateParas();

	//forward
	void forward(double**** x);
	void setInLayerOutput(double**** x);
	void setConvOutput(Layer& layer, Layer& lastLayer);
	void setSampOutput(Layer& layer, Layer& lastLayer);
	void setFullyConnectedHiddenLayerOutput(Layer& layer, Layer& lastLayer);
	void setOutLayerOutput(Layer& layer, Layer& lastLayer);

	//load parameter
	void loadParas();
	void loadBias(Layer& layer, char* str_File_Bias);
	void loadKernels(Layer& layer, Layer& lastLayer, char* str_File_Kernel);
};

