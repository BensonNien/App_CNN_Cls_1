// App_CNN_Cls_1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
/******************************************************************************
Date:  2022/09
Author: CHU-MIN, NIEN
Description:
******************************************************************************/

#include <iostream>
#include "CNNDataStruct.h"
#include "CNNDataset.h"
//#include "CNNCls.h"
#include "CPUCNNCls.h"

using namespace std;

void AIFlowTest();
void AIFlowV1();

int main()
{
	std::cout << "\n====================== Prog. Start ======================\n";
	// initialize input data
	float**** train_x, **** test_x;
	float** train_label, ** test_label;
	int NumberOfImages = 660;//Total training data//nu 5850
	int NumOfChannel = 3;// image's channel

	train_x = new float*** [NumberOfImages];
	test_x = new float*** [NumberOfImages];
	train_label = new float* [NumberOfImages];
	test_label = new float* [NumberOfImages];
	for (int i = 0; i < NumberOfImages; i++)
	{
		train_x[i] = new float** [NumOfChannel];
		test_x[i] = new float** [NumOfChannel];
		for (int c = 0; c < NumOfChannel; c++)
		{
			train_x[i][c] = new float* [IMG_H];
			test_x[i][c] = new float* [IMG_H];

			for (int j = 0; j < IMG_H; j++)
			{
				train_x[i][c][j] = new float[IMG_W];
				test_x[i][c][j] = new float[IMG_W];
			}
		}
		train_label[i] = new float[2];
		test_label[i] = new float[2];
	}


	CNNDataset::Load(train_x, test_x, train_label, test_label);//load data & label

	// constructor CPUCNN
	CPUCNNLayerCreater layer_creater;
	CPUCNNLayer layer;
	layer_creater.AddLayer(layer.CreateInputLayer(3, RectSize(IMG_H, IMG_W)));//image's channel, image RectSize

	layer_creater.AddLayer(layer.CreateConvLayer(3, 6, RectSize(5, 5)));//convolutional layer output feature map's deep number, kernel RectSize
	layer_creater.AddLayer(layer.CreateSampLayer(RectSize(2, 2)));//Downsampling layer kernel RectSize
	layer_creater.AddLayer(layer.CreateConvLayer(6, 12, RectSize(5, 5)));//convolutional layer output feature map's deep number, kernel RectSize
	layer_creater.AddLayer(layer.CreateSampLayer(RectSize(2, 2)));//Downsampling layer kernel RectSize
	layer_creater.AddLayer(layer.CreateConvLayer(12, 20, RectSize(4, 4)));//convolutional layer output feature map's deep number, kernel RectSize
	layer_creater.AddLayer(layer.CreateSampLayer(RectSize(2, 2)));//Downsampling layer kernel RectSize

	//layer_creater.AddLayer(layer.CreateFullyConnectedHiddenLayer(20));//Fully connected hidden layer node number
	//layer_creater.AddLayer(layer.CreateFullyConnectedHiddenLayer(20));//Fully connected hidden layer node number
	layer_creater.AddLayer(layer.CreateOutputLayer(2));//output layer node number

	CPUCNN cnn = CPUCNN(layer_creater, 2);// batchsize

	float t0 = EvlElapsedTime();
	cnn.LoadParas();//load kernel weight & bias

	for (int i = 0; i < 1; i++)//i is training epoch
	{
		cout << "No.of Training: " << i << endl;
		float t1 = EvlElapsedTime();
		cnn.Train(train_x, train_label, NumberOfImages);
		float t2 = EvlElapsedTime();
		cout << t2 - t1 << " s" << endl << i + 1 << endl;
		cout << "No.of Testing: " << i << endl;
		cnn.Inference(test_x, test_label, NumberOfImages);
	}

	float te = EvlElapsedTime();
	cout << "total: " << te - t0 << " s" << endl;

	//for testing
	cnn.LoadParas();//load kernel weight & bias
	cnn.Inference(test_x,test_label, NumberOfImages);


	// delete data
	for (int i = 0; i < NumberOfImages; i++)
	{
		delete[]train_label[i];
		for (int c = 0; c < NumOfChannel; c++)
		{
			for (int j = 0; j < IMG_H; j++)
			{
				delete[]train_x[i][c][j];
			}
			delete[]train_x[i][c];
		}
		delete[]train_x[i];
	}

	for (int i = 0; i < NumberOfImages; i++)
	{
		delete[]test_label[i];
		for (int c = 0; c < NumOfChannel; c++)
		{
			for (int j = 0; j < IMG_H; j++)
			{
				delete[]test_x[i][c][j];
			}
			delete[]test_x[i][c];
		}
		delete[]test_x[i];
	}
	delete[]train_label;
	delete[]train_x;
	delete[]test_x;
	delete[]test_label;
	std::cout << "\n====================== Prog. End ======================\n";
	return 0;
    
}

void AIFlowTest() {
	// constructor CPUCNN
	CPUCNNLayerCreater layer_creater;
	CPUCNNLayer layer;
	layer_creater.AddLayer(layer.CreateInputLayer(3, RectSize(32, 16)));//image's channel, image RectSize

	layer_creater.AddLayer(layer.CreateConvLayer(3, 6, RectSize(5, 5)));//convolutional layer output feature map's deep number, kernel RectSize
	
	CPUCNN cnn = CPUCNN(layer_creater, 2);// batchsize
}

void AIFlowV1() {
}