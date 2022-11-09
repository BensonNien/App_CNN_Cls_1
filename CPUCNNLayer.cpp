/******************************************************************************
Date:  2022/09
Author: CHU-MIN, NIEN
Description:
******************************************************************************/

#include "CPUCNNLayer.h"

using namespace std;

// Utility
float EvlElapsedTime()
{
	return clock() / CLOCKS_PER_SEC;
}

void RandomMatrix(int size_row, int size_col, float* p_kernel)
{
	// construct a trivial random generator engine from a time-based seed:
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<float> distribution(0, 0.14);

	cout << "--------------- kernel's content -----------------" << endl;
	for (int i = 0; i < size_row; i++)
	{
		for (int j = 0; j < size_col; j++)
		{
			p_kernel[i * size_col + j] = distribution(generator);
			cout << to_string(p_kernel[i * size_col + j]) << " ";
		}
		cout << endl;
	}
}

void ConvNValid(float* p_matrix, float* p_kernel, int map_size_row, int map_size_col, int kernel_size_row, int kernel_size_col, float* outmatrix)
{

	// the number of row of convolution
	int num_conv_row = map_size_row - kernel_size_row + 1;
	// the number of column of convolution
	int num_conv_col = map_size_col - kernel_size_col + 1;

	for (int i = 0; i < num_conv_row; i++)
	{
		for (int j = 0; j < num_conv_col; j++)
		{
			float sum = 0.0;
			for (int ki = 0; ki < kernel_size_row; ki++)
			{
				for (int kj = 0; kj < kernel_size_col; kj++)
				{
					//sum += p_matrix[i + ki][j + kj] * p_kernel[ki][kj];
					sum += p_matrix[((i + ki) * map_size_col) + (j + kj)] * p_kernel[(ki * kernel_size_col) + kj];
				}
			}
			outmatrix[(i * num_conv_col) + j] = sum;
		}
	}
}

void ActiveRelu(float* p_matrix, float bias, int m, int n)
{

	float x1 = 0.0;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			x1 = ACTTIVE_RELU(p_matrix[(i * n) + j] + bias);

			if (x1 > 0.0) {
				p_matrix[(i * n) + j] = x1;
			}
			else if (0.0 == x1 || x1 < 0.0) {

				p_matrix[(i * n) + j] = 0.0;

			}
			else {
				exit(0);
			}
		}
	}
}

void ActiveSigmoid(float** matrix, float bias_, int m, int n)
{


	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			matrix[i][j] = ACTTIVE_SIGMOID(matrix[i][j] + bias_);
		}
	}
}

void CalExpone(float* p_matrix, float bias, int m, int n)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			//cout << "Outputlayer's actual ouput(p_matrix[" << i << "][" << j << "] + bias_): " << p_matrix[i][j] + bias_ << endl;
			p_matrix[(i * n) + j] = exp(p_matrix[(i*n)+j] + bias);
			//cout << "Outputlayer's expone actual ouput: " << p_matrix[i][j] << endl;
		}
	}
}

void CalConvArrayPlus(float* x, float* y, int m, int n)
{

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			y[(i * n) + j] += x[(i * n) + j];
		}
	}
}

void CalFCHArrayPlus(float* x, float* y, int m, int n)
{

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			y[(i * n) + j] += x[(i * n) + j];
		}
	}
}

void CalSampArrayPlus(float* x, float* y, int m, int n)
{

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			y[(i * n) + j] += x[(i * n) + j];
		}
	}
}

void CalArrayPlus(float* x, float* y, int m, int n)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			y[i * n + j] += x[i * n + j];
		}
	}
}

void ScaleMatrix(float* p_matrix, RectSize scale, int matrix_rows, int matrix_cols, float* p_out_matrix)
{
	int out_matrix_rows = matrix_rows / scale.x;
	int out_matrix_cols = matrix_cols / scale.y;
	if (out_matrix_rows * scale.x != matrix_rows || out_matrix_cols * scale.y != matrix_cols)
	{
		cout << "scale can not divide by p_matrix";
	}

	float whole_s = (float)(scale.x * scale.y);
	float sum = 0.0;
	for (int i = 0; i < out_matrix_rows; i++) {
		for (int j = 0; j < out_matrix_cols; j++) {
			sum = 0.0;
			for (int si = i * scale.x; si < (i + 1) * scale.x; si++) {
				for (int sj = j * scale.y; sj < (j + 1) * scale.y; sj++) {
					sum += p_matrix[(si * matrix_cols) + sj];
				}
			}
			p_out_matrix[(i * out_matrix_cols) + j] = sum / whole_s;
		}
	}
}

void Rot180(float* p_matrix, int m, int n, float* p_rot_matrix)
{

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			p_rot_matrix[i*n+j] = p_matrix[i*n+(n - 1 - j)];
		}
	}

	for (int j = 0; j < n; j++)
	{
		for (int i = 0; i < m / 2; i++)
		{
			/*float tmp = p_rot_matrix[i][j];
			p_rot_matrix[i][j] = p_rot_matrix[m - 1 - i][j];
			p_rot_matrix[m - 1 - i][j] = tmp;*/
			std::swap(p_rot_matrix[i * n + j], p_rot_matrix[(m - 1 - i)*n+j]);
		}
	}
}

void ConvNSampFull(float* p_matrix, float* p_kernel, int m, int n, int km, int kn, float* p_out_matrix, float* p_extend_matrix)
{

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			//p_extend_matrix[i + km - 1][j + kn - 1] = p_matrix[i][j];
			p_extend_matrix[((i + km - 1)*n)+(j + kn - 1)] = p_matrix[i*n+j];
		}	
	}

	ConvNValid(p_extend_matrix, p_kernel, (m + 2 * (km - 1)), (n + 2 * (kn - 1)), km, kn, p_out_matrix);

}

//void ConvNFull(float** matrix, float** kernel_, int m, int n, int km, int kn, float** outmatrix, float** extendMatrix)
//{
//
//	for (int i = 0; i < m; i++) {
//		for (int j = 0; j < n; j++)
//			extendMatrix[i + km - 1][j + kn - 1] = matrix[i][j];
//	}
//
//	ConvNValid(extendMatrix, kernel_, m + 2 * (km - 1), n + 2 * (kn - 1), km, kn, outmatrix);
//
//}

void MatrixDrelu(float** matrix, int m, int n, float** M)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{

			if (matrix[i][j] > 0.0)
			{
				M[i][j] = 1.0;
			}
			else if (0.0 == matrix[i][j] || matrix[i][j] < 0.0) {
				M[i][j] = 0.0;
			}
		}
	}
}

//for derivation of ReLU active fun. 
void MatrixDreluFChidden(float* matrix, int m, int n, float* M)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{

			if (matrix[i * n + j] > 0.0)
			{
				*M = 1.0;
			}
			else if (0.0 == matrix[i * n + j] || matrix[i * n + j] < 0.0) {
				*M = 0.0;
			}

		}
	}
}

void MatrixDreluConv(float* matrix, int m, int n, float* M)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{

			if (matrix[i * n + j] > 0.0)
			{
				*M = 1.0;
			}
			else if (0.0 == matrix[i * n + j] || matrix[i * n + j] < 0.0) {
				*M = 0.0;
			}

		}
	}
}

void MatrixDsigmoid(float** matrix, int m, int n, float** M)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			M[i][j] = matrix[i][j] * (1 - matrix[i][j]);
		}
	}
}

//for derivation of sigmoid active fun.
void MatrixDsigmoidFChidden(float** matrix, int m, int n, float* M)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			*M = matrix[i][j] * (1 - matrix[i][j]);
		}
	}
}

void Kronecker(float** matrix, RectSize scale, int m, int n, float** OutMatrix)
{
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			for (int ki = i * scale.x; ki < (i + 1) * scale.x; ki++) {
				for (int kj = j * scale.y; kj < (j + 1) * scale.y; kj++) {
					OutMatrix[ki][kj] = matrix[i][j];
				}
			}
		}
	}
}

void CalKronecker(float* p_nextlayer_matrix, RectSize scale, int nextlayer_matrix_rows, int nextlayer_matrix_cols, float* p_out_matrix, int layer_out_matrix_rows, int layer_out_matrix_cols)
{
	for (int i = 0; i < nextlayer_matrix_rows; i++) {
		for (int j = 0; j < nextlayer_matrix_cols; j++) {
			for (int ki = (i * scale.x); ki < ((i + 1) * scale.x); ki++) {
				for (int kj = (j * scale.y); kj < ((j + 1) * scale.y); kj++) {
					p_out_matrix[ki * layer_out_matrix_cols + kj] = p_nextlayer_matrix[i * nextlayer_matrix_cols + j];
				}
			}
		}
	}
}

void MatrixMultiply(float** matrix1, float** matrix2, int m, int n)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			matrix1[i][j] = matrix1[i][j] * matrix2[i][j];
		}
	}
}

void CalMatrixMultiply(float* matrix1, float* matrix2, int m, int n)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			matrix1[i * n + j] = matrix1[i * n + j] * matrix2[i * n + j];
		}
	}
}

//void Sum(float**** errors_, int j, int m, int n, int batchSize, float** M)
//{
//	for (int mi = 0; mi < m; mi++) {
//		for (int nj = 0; nj < n; nj++) {
//			float sum = 0;
//			for (int i = 0; i < batchSize; i++) {
//				sum += errors_[i][j][mi][nj];
//			}
//			M[mi][nj] = sum;
//		}
//	}
//}

//float Sum(float** error, int m, int n)
//{
//	float sum = 0.0;
//	for (int i = 0; i < m; i++) {
//		for (int j = 0; j < n; j++) {
//			sum += error[i][j];
//		}
//	}
//	return sum;
//}

void CalErrorsSum(float* p_errors, int idx_outmap, int outmap_num, int outmap_rows, int outmap_cols, int batch_size, float* p_m)
{
	float sum = 0.0;
	int shift_idx_error_batch_map = 0;
	int shift_idx_error_out_map = 0;
	int idx_error_out_map = 0;
	for (int mi = 0; mi < outmap_rows; mi++) {
		for (int nj = 0; nj < outmap_cols; nj++) {
			sum = 0.0;
			for (int i = 0; i < batch_size; i++) {
				shift_idx_error_batch_map = i * outmap_num * outmap_rows * outmap_cols;
				shift_idx_error_out_map = idx_outmap * outmap_rows * outmap_cols;
				idx_error_out_map = shift_idx_error_batch_map + shift_idx_error_out_map + (mi * outmap_cols) + nj;
				//sum += p_errors[i][j][mi][nj];
				sum += p_errors[idx_error_out_map];
			}
			p_m[mi * outmap_cols + nj] = sum;
		}
	}
}

float CalErrorSum(float* error, int m, int n)
{
	float sum = 0.0;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			sum += error[i*n+j];
		}
	}
	return sum;
}

void CalArrayDivide(float* M, int batchSize, int m, int n)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			M[i * n + j] = M[i * n + j] / batchSize;
		}
	}
}

void CalArrayMultiply(float* matrix, float val, int m, int n)
{

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			matrix[i*n+j] *= val;
		}
	}
}

int findIndex(float*** p)
{

	FILE* fy;
	fy = fopen("./outputdata/outputmaps_.txt", "a");
	/*
	if( (err=fopen_s(&fy, "outputmaps_.txt", "a")) != 0 )
		exit(1) ;
	*/
	int index = 0;
	float v;
	float Max = p[0][0][0];
	fprintf(fy, "%f ", Max);
	for (int i = 1; i < 2; i++)
	{
		v = p[i][0][0];
		fprintf(fy, "%f\n", v);
		if (p[i][0][0] > Max)
		{
			Max = p[i][0][0];
			index = i;
		}
	}
	fclose(fy);
	return index;
}

int FindIndex(float* p_batch_maps, int map_num, int map_rows, int map_cols)
{

	FILE* fy;
	fy = fopen("./outputdata/outputmaps_.txt", "a");
	/*
	if( (err=fopen_s(&fy, "outputmaps_.txt", "a")) != 0 )
		exit(1) ;
	*/
	int shift_idx_layer_out_map = 0 * map_rows * map_cols;
	int idx_layer_out_map = shift_idx_layer_out_map + (0 * map_cols + 0);
	int index = 0;
	float v;
	float Max = p_batch_maps[idx_layer_out_map];
	fprintf(fy, "%f ", Max);
	for (int i = 1; i < map_num; i++)
	{
		shift_idx_layer_out_map = i * map_rows * map_cols;
		idx_layer_out_map = shift_idx_layer_out_map + (0 * map_cols + 0);
		v = p_batch_maps[idx_layer_out_map];
		fprintf(fy, "%f\n", v);
		if (p_batch_maps[idx_layer_out_map] > Max)
		{
			Max = p_batch_maps[idx_layer_out_map];
			index = i;
		}
	}
	fclose(fy);
	return index;
}

int FindIndex(float* p)
{
	int index = 0;
	float Max = p[0];
	for (int i = 1; i < 2; i++)
	{
		float v = p[i];
		if (p[i] > Max)
		{
			Max = p[i];
			index = i;
		}
	}
	return index;
}

void SetInLayerValue(float* maps, float** sum, int m, int n)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			maps[i*n+j] = sum[i][j];
		}
	}
}

void SetKernelValue(float* maps, float* sum, int m, int n)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			maps[i * n + j] = sum[i * n + j];
		}
	}
}

CPUCNNLayer CPUCNNLayer::CreateInputLayer(int input_map_num, RectSize map_size)
{
	CPUCNNLayer layer;
	layer.layer_type_ = 'I';
	layer.in_map_num_ = input_map_num;
	layer.out_map_num_= input_map_num;
	layer.map_size_ = map_size;
	return layer;
}
CPUCNNLayer CPUCNNLayer::CreateConvLayer(int input_map_num, int output_map_num, RectSize kernel_size)
{
	CPUCNNLayer layer;
	layer.layer_type_ = 'C';
	layer.in_map_num_ = input_map_num;
	layer.out_map_num_ = output_map_num;
	layer.kernel_size_ = kernel_size;
	return layer;
}
CPUCNNLayer CPUCNNLayer::CreateSampLayer(RectSize scale_size)
{
	CPUCNNLayer layer;
	layer.layer_type_ = 'S';
	layer.scale_size_ = scale_size;
	return layer;
}
CPUCNNLayer CPUCNNLayer::CreateFullyConnectedHiddenLayer(int input_element_num, int output_element_num, int class_num)
{
	CPUCNNLayer layer;
	layer.in_element_num_ = input_element_num;
	layer.out_element_num_ = output_element_num;
	layer.class_num_ = class_num;
	layer.layer_type_ = 'H';
	layer.map_size_ = RectSize(1, 1);
	layer.out_map_num_ = output_element_num;
	return layer;

}
CPUCNNLayer CPUCNNLayer::CreateOutputLayer(int class_num)
{
	CPUCNNLayer layer;
	//layer.in_element_num_ = input_element_num;
	//layer.out_element_num_ = output_element_num;
	layer.class_num_ = class_num;
	layer.layer_type_ = 'O';
	layer.map_size_ = RectSize(1, 1);
	layer.out_map_num_ = class_num;
	return layer;

}

void CPUCNNLayer::InitKernel(int front_map_num) {
	vec_kernel_.reserve(front_map_num * out_map_num_ * kernel_size_.x * kernel_size_.y);
	vec_kernel_.resize(front_map_num * out_map_num_ * kernel_size_.x * kernel_size_.y);
	int shift_idx_front_map = 0;
	int shift_idx_out_map = 0;

	for (int i = 0; i < front_map_num; i++)
	{
		shift_idx_front_map = i * out_map_num_ * kernel_size_.x * kernel_size_.y;
		for (int j = 0; j < out_map_num_; j++)
		{
			shift_idx_out_map = j * kernel_size_.x * kernel_size_.y;
			RandomMatrix(kernel_size_.x, kernel_size_.y, (vec_kernel_.data() + shift_idx_front_map + shift_idx_out_map));
		}
	}
}
//for adding momentum
void CPUCNNLayer::InitLastStepDeltaKernel(int front_map_num)
{
	vec_laststep_delta_kernel_.reserve(front_map_num * out_map_num_ * kernel_size_.x * kernel_size_.y);
	vec_laststep_delta_kernel_.resize(front_map_num * out_map_num_ * kernel_size_.x * kernel_size_.y);
	vec_laststep_delta_kernel_.assign(vec_laststep_delta_kernel_.size(), 0.0);

	int shift_idx_front_map = 0;
	int shift_idx_out_map = 0;

	/*for (int i = 0; i < front_map_num; i++)
	{
		shift_idx_front_map = i * out_map_num_ * kernel_size_.x * kernel_size_.y;
		for (int j = 0; j < out_map_num_; j++)
		{
			shift_idx_out_map = j * kernel_size_.x * kernel_size_.y;
			for (int ii = 0; ii < kernel_size_.x; ii++)
			{
				for (int jj = 0; jj < kernel_size_.y; jj++)
				{
					vec_laststep_delta_kernel_[shift_idx_front_map + shift_idx_out_map + i * kernel_size_.y + j] = 0.0;
				}
			}
		}
	}*/

}
void CPUCNNLayer::InitOutputKernel(int front_map_num, RectSize Kernel_size)
{
	kernel_size_ = Kernel_size;
	vec_kernel_.reserve(front_map_num * out_map_num_ * kernel_size_.x * kernel_size_.y);
	vec_kernel_.resize(front_map_num * out_map_num_ * kernel_size_.x * kernel_size_.y);
	int shift_idx_front_map = 0;
	int shift_idx_out_map = 0;

	for (int i = 0; i < front_map_num; i++)
	{
		shift_idx_front_map = i * out_map_num_ * kernel_size_.x * kernel_size_.y;
		for (int j = 0; j < out_map_num_; j++)
		{
			shift_idx_out_map = j * kernel_size_.x * kernel_size_.y;
			RandomMatrix(kernel_size_.x, kernel_size_.y, (vec_kernel_.data() + shift_idx_front_map + shift_idx_out_map));
		}
	}

}
//for adding momentum
void CPUCNNLayer::InitOutputLastStepDeltaKernel(int front_map_num, RectSize Kernel_size)
{
	kernel_size_ = Kernel_size;
	vec_laststep_delta_kernel_.reserve(front_map_num * out_map_num_ * kernel_size_.x * kernel_size_.y);
	vec_laststep_delta_kernel_.resize(front_map_num * out_map_num_ * kernel_size_.x * kernel_size_.y);
	vec_laststep_delta_kernel_.assign(vec_laststep_delta_kernel_.size(), 0.0);
}
void CPUCNNLayer::InitErros(int batch_size)
{
	vec_errors_.reserve(batch_size * out_map_num_ * map_size_.x * map_size_.y);
	vec_errors_.resize(batch_size * out_map_num_ * map_size_.x * map_size_.y);
	vec_errors_.assign(vec_errors_.size(), 0.0);
}
void CPUCNNLayer::InitOutputMaps(int batch_size)
{
	vec_output_maps_.reserve(batch_size * out_map_num_ * map_size_.x * map_size_.y);
	vec_output_maps_.resize(batch_size * out_map_num_ * map_size_.x * map_size_.y);
	vec_output_maps_.assign(vec_errors_.size(), 0.0);
}
void CPUCNNLayer::InitBias(int front_map_num, int idx_iter)
{
	vec_bias_.reserve(out_map_num_);
	vec_bias_.resize(out_map_num_);
	vec_bias_.assign(vec_bias_.size(), 0.1);
}
void CPUCNNLayer::SetError(int num_batch, int map_no, int map_x, int map_y, float error_val)
{
	int shift_idx_error_batch_map = num_batch * out_map_num_ * map_size_.x * map_size_.y;
	int shift_idx_error_out_map = map_no * map_size_.x * map_size_.y;
	vec_errors_[shift_idx_error_batch_map + shift_idx_error_out_map + (map_x * map_size_.y) + map_y] = error_val;
}
void CPUCNNLayer::SetFCHLayerError(int num_batch, int map_no, float* p_matrix, int m, int n)
{
	int shift_idx_error_batch_map = num_batch * out_map_num_ * map_size_.x * map_size_.y;
	int shift_idx_error_out_map = map_no * map_size_.x * map_size_.y;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			vec_errors_[shift_idx_error_batch_map + shift_idx_error_out_map + (i * map_size_.y) + j] = p_matrix[(i*n)+j];

		}
	}
}
void CPUCNNLayer::SetSampLayerError(int num_batch, int map_no, float* p_matrix, int m, int n)
{
	int shift_idx_error_batch_map = num_batch * out_map_num_ * map_size_.x * map_size_.y;
	int shift_idx_error_out_map = map_no * map_size_.x * map_size_.y;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			vec_errors_[shift_idx_error_batch_map + shift_idx_error_out_map + (i * map_size_.y) + j] = p_matrix[(i * n) + j];

		}
	}
}
void CPUCNNLayer::SetConvLayerError(int num_batch, int map_no, float* p_matrix, int m, int n)
{
	int shift_idx_error_batch_map = num_batch * out_map_num_ * map_size_.x * map_size_.y;
	int shift_idx_error_out_map = map_no * map_size_.x * map_size_.y;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			vec_errors_[shift_idx_error_batch_map + shift_idx_error_out_map + (i * map_size_.y) + j] = p_matrix[(i * n) + j];

		}
	}
}