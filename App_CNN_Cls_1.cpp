// App_CNN_Cls_1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
/******************************************************************************
Date:  2022/09
Author: CHU-MIN, NIEN
Description:
******************************************************************************/

#include <iostream>
#include "CNNDataset.h"
#include "CNNCls.h"

using namespace std;

int main()
{
	std::cout << "\n====================== Prog. Start ======================\n";

	// initialize input data
	double**** train_x, **** test_x;
	double** train_label, ** test_label;
	int NumberOfImages = 660;//Total training data//nu 5850
	int NumOfChannel = 3;// image's channel

	train_x = new double*** [NumberOfImages];
	test_x = new double*** [NumberOfImages];
	train_label = new double* [NumberOfImages];
	test_label = new double* [NumberOfImages];
	for (int i = 0; i < NumberOfImages; i++)
	{
		train_x[i] = new double** [NumOfChannel];
		test_x[i] = new double** [NumOfChannel];
		for (int c = 0; c < NumOfChannel; c++)
		{
			train_x[i][c] = new double* [IMG_H];
			test_x[i][c] = new double* [IMG_H];

			for (int j = 0; j < IMG_H; j++)
			{
				train_x[i][c][j] = new double[IMG_W];
				test_x[i][c][j] = new double[IMG_W];
			}
		}
		train_label[i] = new double[2];
		test_label[i] = new double[2];
	}


	CNNDataset::Load(train_x, test_x, train_label, test_label);//load data & label

	// constructor CNN
	LayerBuilder builder;
	Layer layer;
	builder.addLayer(layer.buildInputLayer(3, RectSize(IMG_H, IMG_W)));//image's channel, image RectSize

	builder.addLayer(layer.buildConvLayer(6, RectSize(5, 5)));//convolutional layer output feature map's deep number, kernel RectSize
	builder.addLayer(layer.buildSampLayer(RectSize(2, 2)));//Downsampling layer kernel RectSize
	builder.addLayer(layer.buildConvLayer(12, RectSize(5, 5)));//convolutional layer output feature map's deep number, kernel RectSize
	builder.addLayer(layer.buildSampLayer(RectSize(2, 2)));//Downsampling layer kernel RectSize
	builder.addLayer(layer.buildConvLayer(20, RectSize(4, 4)));//convolutional layer output feature map's deep number, kernel RectSize
	builder.addLayer(layer.buildSampLayer(RectSize(2, 2)));//Downsampling layer kernel RectSize

	//builder.addLayer(layer.buildFullyConnectedHiddenLayer(20));//Fully connected hidden layer node number
	//builder.addLayer(layer.buildFullyConnectedHiddenLayer(20));//Fully connected hidden layer node number
	builder.addLayer(layer.buildOutputLayer(2));//output layer node number

	CNN cnn = CNN(builder, 2);// batchsize

	double t0 = cpu_time();
	cnn.loadParas();//load kernel weight & bias

	for (int i = 0; i < 1; i++)//i is training epoch
	{
		cout << "No.of Training: " << i << endl;
		double t1 = cpu_time();
		cnn.train(train_x, train_label, NumberOfImages);
		double t2 = cpu_time();
		cout << t2 - t1 << " s" << endl << i + 1 << endl;
		cout << "No.of Testing: " << i << endl;
		cnn.test(test_x, test_label, NumberOfImages);
	}

	double te = cpu_time();
	cout << "total: " << te - t0 << " s" << endl;

	//for testing
	//cnn.loadParas();//load kernel weight & bias
	//cnn.test(test_x,test_label, NumberOfImages);


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
