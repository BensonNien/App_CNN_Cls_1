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
	float**** p_train_images, **** p_test_images;
	float** p_train_labels, ** p_test_labels;
	int num_images = 2000;//Total training data//nu 5850
	int num_pos_images = 1000;
	int num_neg_images = 1000;
	int channels_image = 3;

	std::string pos_images_root_path = ".\\Pedestrian_TrainingDataset_PNG\\64x32_part_balance_v2\\pos\\Training_p_";
	std::string neg_images_root_path = ".\\Pedestrian_TrainingDataset_PNG\\64x32_part_balance_v2\\neg\\Training_n_";
	std::string images_ext = ".png";

	p_train_images = new float*** [num_images];
	p_test_images = new float*** [num_images];
	p_train_labels = new float* [num_images];
	p_test_labels = new float* [num_images];
	for (int i = 0; i < num_images; i++)
	{
		p_train_images[i] = new float** [channels_image];
		p_test_images[i] = new float** [channels_image];
		for (int c = 0; c < channels_image; c++)
		{
			p_train_images[i][c] = new float* [IMG_H];
			p_test_images[i][c] = new float* [IMG_H];

			for (int j = 0; j < IMG_H; j++)
			{
				p_train_images[i][c][j] = new float[IMG_W];
				p_test_images[i][c][j] = new float[IMG_W];
			}
		}
		p_train_labels[i] = new float[2];
		p_test_labels[i] = new float[2];
	}

	DatasetLoadingParamPKG dataset_param(p_train_images, p_test_images, 
		p_train_labels, p_test_labels, num_pos_images, num_neg_images, channels_image, 
		pos_images_root_path, neg_images_root_path, images_ext);

	CNNDataset::Load(dataset_param);

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

	layer_creater.AddLayer(layer.CreateFullyConnectedHiddenLayer(20, 14, 2));//Fully connected hidden layer node number
	layer_creater.AddLayer(layer.CreateOutputLayer(2));//output layer node number

	CPUCNN cnn = CPUCNN(layer_creater, 2);// batchsize

	float t0 = EvlElapsedTime();
	//cnn.LoadParas();//load kernel weight & bias

	for (int i = 0; i < 50; i++)//i is training epoch
	{
		cout << "No.of Training: " << i << endl;
		float t1 = EvlElapsedTime();
		cnn.Train(p_train_images, p_train_labels, num_images);
		float t2 = EvlElapsedTime();
		cout << t2 - t1 << " s" << endl << i + 1 << endl;
		cout << "No.of Testing: " << i << endl;
		cnn.Inference(p_test_images, p_test_labels, num_images);
	}

	float te = EvlElapsedTime();
	cout << "total: " << te - t0 << " s" << endl;

	//for testing
	cnn.LoadParas();//load kernel weight & bias
	cnn.Inference(p_test_images,p_test_labels, num_images);


	// delete data
	for (int i = 0; i < num_images; i++)
	{
		delete[]p_train_labels[i];
		for (int c = 0; c < channels_image; c++)
		{
			for (int j = 0; j < IMG_H; j++)
			{
				delete[]p_train_images[i][c][j];
			}
			delete[]p_train_images[i][c];
		}
		delete[]p_train_images[i];
	}

	for (int i = 0; i < num_images; i++)
	{
		delete[]p_test_labels[i];
		for (int c = 0; c < channels_image; c++)
		{
			for (int j = 0; j < IMG_H; j++)
			{
				delete[]p_test_images[i][c][j];
			}
			delete[]p_test_images[i][c];
		}
		delete[]p_test_images[i];
	}
	delete[]p_train_labels;
	delete[]p_train_images;
	delete[]p_test_images;
	delete[]p_test_labels;
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