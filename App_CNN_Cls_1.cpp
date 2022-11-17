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
#include "CPUCNNCls.h"

using namespace std;

int main()
{
	std::cout << "\n====================== Prog. Start ======================\n";
	// initialize input data
	
	size_t num_pos_train_images = 1000;
	size_t num_neg_train_images = 1000;
	size_t num_train_images = num_pos_train_images + num_neg_train_images;
	size_t num_pos_validation_images = 1000;
	size_t num_neg_validation_images = 1000;
	size_t num_validation_images = num_pos_validation_images + num_neg_validation_images;
	size_t num_pos_test_images = 1000;
	size_t num_neg_test_images = 1000;
	size_t num_test_images = num_pos_test_images + num_neg_test_images;
	size_t rows_image = 64;
	size_t cols_image = 32;
	size_t channels_image = 3;
	size_t num_output_cls = 2;

	std::string pos_train_images_root_path = ".\\Pedestrian_TrainingDataset_PNG\\64x32_part_balance_v2\\pos\\Training_p_";
	std::string neg_train_images_root_path = ".\\Pedestrian_TrainingDataset_PNG\\64x32_part_balance_v2\\neg\\Training_n_";
	std::string pos_validation_images_root_path = pos_train_images_root_path;
	std::string neg_validation_images_root_path = neg_train_images_root_path;
	std::string pos_test_images_root_path = pos_train_images_root_path;
	std::string neg_test_images_root_path = neg_train_images_root_path;
	std::string images_ext = ".png";


	DatasetLoadingParamPKG train_dataset_param(num_pos_train_images, num_neg_train_images,
		rows_image, cols_image, channels_image, num_output_cls,
		pos_train_images_root_path, neg_train_images_root_path,
		images_ext);

	DatasetLoadingParamPKG validation_dataset_param(num_pos_train_images, num_neg_train_images,
		rows_image, cols_image, channels_image, num_output_cls,
		pos_train_images_root_path, neg_train_images_root_path,
		images_ext);

	DatasetLoadingParamPKG test_dataset_param(num_pos_train_images, num_neg_train_images,
		rows_image, cols_image, channels_image, num_output_cls,
		pos_train_images_root_path, neg_train_images_root_path,
		images_ext);

	CNNDataset::Load(train_dataset_param);
	CNNDataset::Load(validation_dataset_param);
	CNNDataset::Load(test_dataset_param);

	// constructor CPUCNN
	CPUCNNLayerCreater layer_creater;
	CPUCNNLayer layer;
	layer_creater.AddLayer(layer.CreateInputLayer(3, RectSize(rows_image, cols_image)));//image's channel, image RectSize

	layer_creater.AddLayer(layer.CreateConvLayer(3, 6, RectSize(5, 5)));//convolutional layer output feature map's deep number, kernel RectSize
	layer_creater.AddLayer(layer.CreateSampLayer(RectSize(2, 2)));//Downsampling layer kernel RectSize
	layer_creater.AddLayer(layer.CreateConvLayer(6, 12, RectSize(5, 5)));//convolutional layer output feature map's deep number, kernel RectSize
	layer_creater.AddLayer(layer.CreateSampLayer(RectSize(2, 2)));//Downsampling layer kernel RectSize
	layer_creater.AddLayer(layer.CreateConvLayer(12, 20, RectSize(4, 4)));//convolutional layer output feature map's deep number, kernel RectSize
	layer_creater.AddLayer(layer.CreateSampLayer(RectSize(2, 2)));//Downsampling layer kernel RectSize

	layer_creater.AddLayer(layer.CreateFullyConnectedHiddenLayer(20, 14, num_output_cls));//Fully connected hidden layer node number
	layer_creater.AddLayer(layer.CreateOutputLayer(num_output_cls));//output layer node number

	CPUCNN cnn = CPUCNN(layer_creater, 2);// batchsize

	float t0 = EvlElapsedTime();
	//cnn.LoadParas();//load kernel weight & bias

	for (size_t i = 0; i < 50; i++)//i is training epoch
	{
		cout << "No.of Training: " << i << endl;
		float t1 = EvlElapsedTime();
		cnn.Train(train_dataset_param);
		float t2 = EvlElapsedTime();
		cout << t2 - t1 << " s" << endl << i + 1 << endl;
		cout << "No.of Testing: " << i << endl;
		cnn.Inference(validation_dataset_param);
	}

	float te = EvlElapsedTime();
	cout << "total: " << te - t0 << " s" << endl;

	//for testing
	cnn.LoadParas();//load kernel weight & bias
	cnn.Inference(test_dataset_param);

	std::cout << "\n====================== Prog. End ======================\n";
	return 0;
    
}
