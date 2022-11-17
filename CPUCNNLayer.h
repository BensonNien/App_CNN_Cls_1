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

#define ACTTIVE_RELU(x) x //the relu active function
#define ACTTIVE_SIGMOID(x) (1/(1+exp(-x))) //the sigmoid active function

using namespace std;
using namespace cv;

// Utility
float EvlElapsedTime();
void RandomMatrix(int size_row, int size_col, float* p_kernel);
void ConvNValid(float* p_matrix, float* p_kernel, int map_size_row, int map_size_col, int kernel_size_row, int kernel_size_col, float* outmatrix);// m n is the dimension of matrix and km kn is the dimension of kernel_, outmatrix is result
void ActiveRelu(float* p_matrix, float bias, int m, int n);// m n is the dimension of matrix
void ActiveSigmoid(float** matrix, float bias_, int m, int n);// m n is the dimension of matrix
void CalExpone(float* p_matrix, float bias, int m, int n);// m n is the dimension of matrix
void CalConvArrayPlus(float* x, float* y, int m, int n);
void CalFCHArrayPlus(float* x, float* y, int m, int n);
void CalSampArrayPlus(float* x, float* y, int m, int n);
void CalArrayPlus(float* x, float* y, int m, int n);
void ScaleMatrix(float* p_matrix, RectSize scale, int matrix_rows, int matrix_cols, float* p_out_matrix);//sampling
void Rot180(float* p_matrix, int m, int n, float* p_rot_matrix);
void ConvNSampFull(float* p_matrix, float* p_kernel, int m, int n, int km, int kn, float* p_out_matrix, float* p_extend_matrix);// convn full mode
//void ConvNFull(float** matrix, float** kernel_, int m, int n, int km, int kn, float** outmatrix, float** extendMatrix);// convn full mode
void MatrixDrelu(float** matrix, int m, int n, float** M);// calculate derivation of ReLU function with matrix
void MatrixDreluFChidden(float* matrix, int m, int n, float* M);// calculate derivation of ReLU function in FChiddenlayer with matrix
void MatrixDreluConv(float* matrix, int m, int n, float* M);// calculate derivation of ReLU function in Convlayer with matrix
void MatrixDsigmoid(float** matrix, int m, int n, float** M);// calculate derivation of sigmoid function with matrix
void MatrixDsigmoidFChidden(float** matrix, int m, int n, float* M);// calculate derivation of sigmoid function in FChiddenlayer with matrix
void Kronecker(float** matrix, RectSize scale, int m, int n, float** outmatrix);
void CalKronecker(float* p_nextlayer_matrix, RectSize scale, int nextlayer_matrix_rows, int nextlayer_matrix_cols, float* p_out_matrix, int layer_out_matrix_rows, int layer_out_matrix_cols);
void MatrixMultiply(float** matrix1, float** matrix2, int m, int n);//inner product of matrix 1 and matrix 2, result is matrix1
void CalMatrixMultiply(float* matrix1, float* matrix2, int m, int n);
//void Sum(float**** errors_, int j, int m, int n, int batchSize, float** outmatrix);
//float CalSum(float** error, int m, int n);
void CalErrorsSum(float* p_errors, int idx_outmap, int outmap_num, int outmap_rows, int outmap_cols, int batch_size, float* p_m);
float CalErrorSum(float* error, int m, int n);
void CalArrayDivide(float* matrix, int batchSize, int m, int n);// result is matrix;
void CalArrayMultiply(float* matrix, float val, int m, int n);// array multiply a float value, result in matrix
void SetInLayerValue(float* maps, float** sum, int m, int n);
void SetKernelValue(float* maps, float* sum, int m, int n);
int findIndex(float*** p);
size_t FindIndex(float* p_batch_maps, size_t map_num, size_t map_rows, size_t map_cols);
size_t FindIndex(float* p_batch_labels, size_t map_num);

// CPUCNNLayer

class CPUCNNLayer
{
private:
	int in_map_num_;
	int out_map_num_;
	char layer_type_;
	RectSize map_size_;
	RectSize scale_size_;
	RectSize kernel_size_;
	int in_element_num_;
	int out_element_num_;
	int class_num_;

public:
	CPUCNNLayer() {};
	~CPUCNNLayer() {};

	vector<float> vec_kernel_;
	vector<float> vec_laststep_delta_kernel_;//for adding momentum
	vector<float> vec_output_maps_;
	vector<float> vec_errors_;
	vector<float> vec_bias_;

	CPUCNNLayer CreateInputLayer(int input_map_num, RectSize map_size);
	CPUCNNLayer CreateConvLayer(int input_map_num, int output_map_num, RectSize kernel_size);
	CPUCNNLayer CreateSampLayer(RectSize scale_size);
	CPUCNNLayer CreateFullyConnectedHiddenLayer(int input_element_num, int output_element_num, int class_num);
	//CPUCNNLayer CreateOutputLayer(int input_element_num, int output_element_num, int class_num);
	CPUCNNLayer CreateOutputLayer(int class_num);

	void InitKernel(int front_map_num);
	void InitLastStepDeltaKernel(int front_map_num);//for adding momentum
	void InitOutputKernel(int front_map_num, RectSize Kernel_size);
	void InitOutputLastStepDeltaKernel(int front_map_num, RectSize Kernel_size);//for adding momentum
	void InitErros(int batch_size);
	void InitOutputMaps(int batch_size);
	void InitBias(int front_map_num, int idx_iter);

	void SetError(int num_batch, int map_no, int map_x, int map_y, float error_val);
	float* GetError(int num_batch, int map_no) {
		int shift_idx_error_batch_map = num_batch * out_map_num_ * map_size_.x * map_size_.y;
		int shift_idx_error_out_map = map_no * map_size_.x * map_size_.y;
		return (vec_errors_.data() + shift_idx_error_batch_map + shift_idx_error_out_map);
	}
	void SetFCHLayerError(int num_batch, int map_no, float* p_matrix, int m, int n);
	void SetSampLayerError(int num_batch, int map_no, float* p_matrix, int m, int n);
	void SetConvLayerError(int num_batch, int map_no, float* p_matrix, int m, int n);
	float* GetKernel(int num_batch, int map_no) {
		int shift_idx_front_map = num_batch * out_map_num_ * kernel_size_.x * kernel_size_.y;
		int shift_idx_out_map = map_no * kernel_size_.x * kernel_size_.y;
		return (vec_kernel_.data() + shift_idx_front_map + shift_idx_out_map);
	}
	int GetOutMapNum() {
		return out_map_num_;
	}
	char GetType() {
		return layer_type_;
	}
	RectSize GetMapSize() {
		return map_size_;
	}
	void SetMapSize(RectSize map_size) {
		this->map_size_ = map_size;
	}
	void SetOutMapNum(int out_map_num) {
		this->out_map_num_ = out_map_num;
	}
	RectSize GetKernelSize() {
		return kernel_size_;
	}
	RectSize GetScaleSize() {
		return scale_size_;
	}

};