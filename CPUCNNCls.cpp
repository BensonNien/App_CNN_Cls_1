/******************************************************************************
Date:  2022/09
Author: CHU-MIN, NIEN
Description:
******************************************************************************/

#include "CPUCNNCls.h"
#include "CPUCNNLayer.h"

using namespace std;

// CPUCNN
#define SELECT_ACTIVE_FUNCTION 3 
//'1' : active function is "sigmoid" , cost function is "quadratic" for "sigmoid" form
//'2' : active function is "sigmoid" , cost function is "Cross entropy" for "sigmoid" form
//'3' : active function is "ReLU" , cost function is "Cross entropy" for "softmax" form

#define DERIV_ACTIVE_RELU(S) 1 // derivative of the relu as a function of the relu's output
#define DERIV_ACTIVE_SIGMOID(S) (S*(1-S)) // derivative of the sigmoid as a function of the sigmoid's output

int g_input_num_channel = 3;//image's channel

int g_idx_epoch = 0;//index of epoch
int g_idx_itor = 0;//index of iterator
int g_idx_iter_init_bias = 0;//index of iterator for initialize bias
int g_idx_iteration_num = 0;//index of iteration
int g_iteration_num = 0;//number of g_iteration_num

void CPUCNN::Train(float**** train_x, float** train_label, int NumOfImage)
{
	std::cout << "Start train" << std::endl;

	g_iteration_num = NumOfImage / batch_size_;
	if (NumOfImage % batch_size_ != 0)
	{
		std::cout << "Please reset batch_size_!" << std::endl;
	}
	float**** Train;
	float** TrainLabel;

	Train = new float*** [batch_size_];
	TrainLabel = new float* [batch_size_];

	for (g_idx_iteration_num = 0; g_idx_iteration_num < g_iteration_num; g_idx_iteration_num++)
	{

		std::cout << "NO.of iteration(training): " << g_idx_iteration_num << std::endl;
		int ii = g_idx_iteration_num % (NumOfImage / batch_size_);
		for (int j = 0; j < batch_size_; j++)
		{

			std::cout << "NO.of batch(training): " << j << std::endl;

			if (g_idx_iteration_num == 0)
			{
				Train[j] = new float** [g_input_num_channel];
			}
			for (int c = 0; c < g_input_num_channel; c++)
			{

				if (g_idx_iteration_num == 0)
				{
					Train[j][c] = new float* [IMG_H];
				}
				for (int m = 0; m < IMG_H; m++)
				{
					if (g_idx_iteration_num == 0)
					{
						Train[j][c][m] = new float[IMG_W];
					}
					for (int n = 0; n < IMG_W; n++)
					{
						Train[j][c][m][n] = train_x[ii * batch_size_ + j][c][m][n];

					}
				}
				if (g_idx_iteration_num == 0)
				{
					TrainLabel[j] = new float[2];
				}
				for (int l = 0; l < 2; l++)
				{
					TrainLabel[j][l] = train_label[ii * batch_size_ + j][l];
				}
			}
		}


		Forward(Train);
		BackPropagation(Train, TrainLabel);
		UpdateParas();


	}
	std::cout << "Finish train" << std::endl;

	for (int i = 0; i < batch_size_; i++)
	{
		delete[]TrainLabel[i];
		for (int c = 0; c < g_input_num_channel; c++)
		{
			for (int j = 0; j < IMG_H; j++)
			{
				delete[]Train[i][c][j];
			}
			delete[]Train[i][c];
		}
		delete[]Train[i];
	}
	delete[]Train;
	delete[]TrainLabel;
}

void CPUCNN::Setup(int batch_size_)
{
	VECCPUCNNLayers::iterator iter = vec_layers_.begin();

	(*iter).InitOutputMaps(batch_size_);
	iter++;
	for (iter; iter < vec_layers_.end(); iter++)
	{
		g_idx_iter_init_bias = g_idx_iter_init_bias + 1;

		int frontMapNum = (*(iter - 1)).GetOutMapNum();

		switch ((*iter).GetType())
		{
		case 'I':
			break;
		case 'C':
			// set map RectSize
			(*iter).SetMapSize((*(iter - 1)).GetMapSize().substract((*iter).GetKernelSize(), 1));
			// initial convolution kernel_, quantities: frontMapNum*outMapNum_
			(*iter).InitKernel(frontMapNum);
			(*iter).InitLastStepDeltaKernel(frontMapNum);//for adding momentum
			//each map has one bias_, so frontMapNum is not necessary
			(*iter).InitBias(frontMapNum, g_idx_iter_init_bias);
			(*iter).InitErros(batch_size_);
			// each layer should initialize output map
			(*iter).InitOutputMaps(batch_size_);
			break;
		case 'S':
			(*iter).SetOutMapNum((frontMapNum));
			(*iter).SetMapSize((*(iter - 1)).GetMapSize().divide((*iter).GetScaleSize()));
			(*iter).InitErros(batch_size_);
			(*iter).InitOutputMaps(batch_size_);
			break;
		case 'H':
			(*iter).InitOutputKernel(frontMapNum, (*(iter - 1)).GetMapSize());
			(*iter).InitOutputLastStepDeltaKernel(frontMapNum, (*(iter - 1)).GetMapSize());//for adding momentum			
			(*iter).InitBias(frontMapNum, g_idx_iter_init_bias);
			(*iter).InitErros(batch_size_);
			(*iter).InitOutputMaps(batch_size_);
			break;
		case 'O':
			(*iter).InitOutputKernel(frontMapNum, (*(iter - 1)).GetMapSize());
			(*iter).InitOutputLastStepDeltaKernel(frontMapNum, (*(iter - 1)).GetMapSize());//for adding momentum
			(*iter).InitBias(frontMapNum, g_idx_iter_init_bias);
			(*iter).InitErros(batch_size_);
			(*iter).InitOutputMaps(batch_size_);
			break;
		default:
			break;
		}
	}
}

void CPUCNN::SetupTest(int batch_size_)
{
	VECCPUCNNLayers::iterator iter = vec_layers_.begin();

	(*iter).InitOutputMaps(batch_size_);
	iter++;
	for (iter; iter < vec_layers_.end(); iter++)
	{
		g_idx_iter_init_bias = g_idx_iter_init_bias + 1;

		int frontMapNum = (*(iter - 1)).GetOutMapNum();

		switch ((*iter).GetType())
		{
		case 'I':
			break;
		case 'C':
			// set map RectSize
			(*iter).SetMapSize((*(iter - 1)).GetMapSize().substract((*iter).GetKernelSize(), 1));
			// initial convolution kernel_, quantities: frontMapNum*outMapNum_
			(*iter).InitKernel(frontMapNum);

			break;

		default:
			break;
		}
	}
}


void CPUCNN::BackPropagation(float**** x, float** y)
{
	SetOutLayerErrors(x, y);
	SetHiddenLayerErrors();
}

void CPUCNN::Forward(float**** x)
{
	SetInLayerOutput(x);
	VECCPUCNNLayers::iterator iter = vec_layers_.begin();
	iter++;
	for (iter; iter < vec_layers_.end(); iter++)
	{
		switch ((*iter).GetType())
		{
		case 'C':
			SetConvOutput((*iter), (*(iter - 1)));
			break;
		case 'S':
			SetSampOutput((*iter), (*(iter - 1)));
			break;
		case 'H':
			SetFCHLayerOutput((*iter), (*(iter - 1)));
			break;
		case 'O':
			SetOutLayerOutput((*iter), (*(iter - 1)));
			break;
		default:
			break;
		}

	}
}

void CPUCNN::SetInLayerOutput(float**** x)
{
	std::cout << "Execute CPUCNN::SetInLayerOutput()" << std::endl;

	VECCPUCNNLayers::iterator iter = vec_layers_.begin();

	RectSize map_size = (*iter).GetMapSize();
	if (IMG_H != map_size.x)
	{
		std::cout << "IMG_H != mapSize_.x" << std::endl;
	}
	for (int i = 0; i < batch_size_; i++)
	{
		for (int c = 0; c < g_input_num_channel; c++)
		{
			int shift_idx_batch_map = i * g_input_num_channel * map_size.x * map_size.y;
			int shift_idx_out_map = c * map_size.x * map_size.y;
			SetInLayerValue(((*iter).vec_output_maps_.data()+ shift_idx_batch_map+ shift_idx_out_map), x[i][c], IMG_H, IMG_W);
			//memcpy();
		}
	}
}
// for change the value in m_Layers
void CPUCNN::SetConvOutput(CPUCNNLayer& layer, CPUCNNLayer& lastLayer)
{
	std::cout << "Execute CPUCNN::SetConvOutput()" << std::endl;
	
	int mapNum = layer.GetOutMapNum();
	int lastMapNum = lastLayer.GetOutMapNum();
	int lastlayer_map_x = lastLayer.GetMapSize().x;
	int lastlayer_map_y = lastLayer.GetMapSize().y;
	int layer_kernel_x = layer.GetKernelSize().x;
	int layer_kernel_y = layer.GetKernelSize().y;
	int x = layer.GetMapSize().x;
	int y = layer.GetMapSize().y;
	vector<float> vec_sum(x * y, 0.0);
	vector<float> vec_sum_now(x * y, 0.0);

	for (int idx_batch = 0; idx_batch < batch_size_; idx_batch++)
	{
		for (int i = 0; i < mapNum; i++)
		{
			for (int j = 0; j < lastMapNum; j++)
			{
				int shift_idx_lastlayer_batch_map = idx_batch * lastMapNum * lastlayer_map_x * lastlayer_map_y;
				int shift_idx_lastlayer_out_map = j * lastlayer_map_x * lastlayer_map_y;
				float* p_lastlayer_map = lastLayer.vec_output_maps_.data() + shift_idx_lastlayer_batch_map + shift_idx_lastlayer_out_map;
				//float** lastMap;
				//lastMap = lastLayer.outputmaps_[idx_batch][j];				
				int shift_idx_layer_front_kernel = j * mapNum * layer_kernel_x * layer_kernel_y;
				int shift_idx_layer_out_kernel = i * layer_kernel_x * layer_kernel_y;
				float* p_layer_kernel = layer.vec_kernel_.data() + shift_idx_layer_front_kernel + shift_idx_layer_out_kernel;

				if (j == 0)
				{
					ConvNValid(p_lastlayer_map, p_layer_kernel, lastlayer_map_x, lastlayer_map_y, layer_kernel_x, layer_kernel_y, vec_sum.data());
					//each time we calculate one image of batch and also calculate relu 

				}
				else {
					ConvNValid(p_lastlayer_map, p_layer_kernel, lastlayer_map_x, lastlayer_map_y, layer_kernel_x, layer_kernel_y, vec_sum_now.data());
					CalConvArrayPlus(vec_sum_now.data(), vec_sum.data(), x, y);// sumNow 

				}
			}

#if((SELECT_ACTIVE_FUNCTION == 1) || (SELECT_ACTIVE_FUNCTION == 2))
			//printf("Logistic sigmoid");
			//Logistic sigmoid
			Sigmoid(sum, layer.bias_[i], x, y);//for sigmoid active fun.
#elif(SELECT_ACTIVE_FUNCTION == 3)
			//printf("ActiveRelu");
			//ActiveRelu
			//ActiveRelu(sum, layer.bias_[i], x, y);//for relu active fun.
			ActiveRelu(vec_sum.data(), layer.vec_bias_.at(i), x, y);//for relu active fun.
#endif

			//SetValue(layer.outputmaps_[idx_batch][i], sum, x, y);
			int shift_idx_layer_batch_map = idx_batch * mapNum * x * y;
			int shift_idx_layer_out_map = i * x * y;
			float* p_layer_out_map = layer.vec_output_maps_.data() + shift_idx_layer_batch_map + shift_idx_layer_out_map;
			memcpy(p_layer_out_map, vec_sum.data(), (x * y * sizeof(float)));

		}
	}

}

void CPUCNN::SetSampOutput(CPUCNNLayer& layer, CPUCNNLayer& lastLayer)
{
	std::cout << "Execute CPUCNN::SetSampOutput()" << std::endl;

	int lastMapNum = lastLayer.GetOutMapNum();
	int lastlayer_map_x = lastLayer.GetMapSize().x;
	int lastlayer_map_y = lastLayer.GetMapSize().y;
	int x = layer.GetMapSize().x;
	int y = layer.GetMapSize().y;
	RectSize scale_size = layer.GetScaleSize();
	vector<float> vec_samp_matrix(x*y, 0.0);

	float* p_lastlayer_map = NULL;
	int shift_idx_lastlayer_batch_map = 0;
	int shift_idx_lastlayer_out_map = 0;

	for (int idx_batch = 0; idx_batch < batch_size_; idx_batch++)
	{
		for (int i = 0; i < lastMapNum; i++)
		{
			//lastMap = lastLayer.outputmaps_[idx_batch][i];
			//ScaleMatrix(lastMap, scale_size, lastlayer_map_x, lastlayer_map_y, vec_samp_matrix);

			//SetValue(layer.outputmaps_[idx_batch][i], vec_samp_matrix, x, y);
			
			shift_idx_lastlayer_batch_map = idx_batch * lastMapNum * lastlayer_map_x * lastlayer_map_y;
			shift_idx_lastlayer_out_map = i * lastlayer_map_x * lastlayer_map_y;
			p_lastlayer_map = lastLayer.vec_output_maps_.data() + shift_idx_lastlayer_batch_map + shift_idx_lastlayer_out_map;			
			ScaleMatrix(p_lastlayer_map, scale_size, lastlayer_map_x, lastlayer_map_y, vec_samp_matrix.data());
			
			int shift_idx_layer_batch_map = idx_batch * lastMapNum * x * y;
			int shift_idx_layer_out_map = i * x * y;
			float* p_layer_out_map = layer.vec_output_maps_.data() + shift_idx_layer_batch_map + shift_idx_layer_out_map;
			memcpy(p_layer_out_map, vec_samp_matrix.data(), (x * y * sizeof(float)));
		}
	}
}

void CPUCNN::SetFCHLayerOutput(CPUCNNLayer& layer, CPUCNNLayer& lastLayer)
{
	std::cout << "Execute CPUCNN::SetFCHLayerOutput()" << std::endl;

	int mapNum = layer.GetOutMapNum();
	int lastMapNum = lastLayer.GetOutMapNum();
	int lastlayer_map_x = lastLayer.GetMapSize().x;
	int lastlayer_map_y = lastLayer.GetMapSize().y;
	int layer_kernel_x = layer.GetKernelSize().x;
	int layer_kernel_y = layer.GetKernelSize().y;
	int x = layer.GetMapSize().x;
	int y = layer.GetMapSize().y;
	vector<float> vec_sum(x * y, 0.0);
	vector<float> vec_sum_now(x * y, 0.0);

	for (int idx_batch = 0; idx_batch < batch_size_; idx_batch++)
	{
		for (int i = 0; i < mapNum; i++)
		{
			for (int j = 0; j < lastMapNum; j++)
			{
				int shift_idx_lastlayer_batch_map = idx_batch * lastMapNum * lastlayer_map_x * lastlayer_map_y;
				int shift_idx_lastlayer_out_map = j * lastlayer_map_x * lastlayer_map_y;
				float* p_lastlayer_map = lastLayer.vec_output_maps_.data() + shift_idx_lastlayer_batch_map + shift_idx_lastlayer_out_map;
				//float** lastMap;
				//lastMap = lastLayer.outputmaps_[idx_batch][j];				
				int shift_idx_layer_front_kernel = j * mapNum * layer_kernel_x * layer_kernel_y;
				int shift_idx_layer_out_kernel = i * layer_kernel_x * layer_kernel_y;
				float* p_layer_kernel = layer.vec_kernel_.data() + shift_idx_layer_front_kernel + shift_idx_layer_out_kernel;

				if (j == 0)
				{
					ConvNValid(p_lastlayer_map, p_layer_kernel, lastlayer_map_x, lastlayer_map_y, layer_kernel_x, layer_kernel_y, vec_sum.data());
					//each time we calculate one image of batch and also calculate relu 

				}
				else {
					ConvNValid(p_lastlayer_map, p_layer_kernel, lastlayer_map_x, lastlayer_map_y, layer_kernel_x, layer_kernel_y, vec_sum_now.data());
					CalFCHArrayPlus(vec_sum_now.data(), vec_sum.data(), x, y);// sumNow 

				}
			}

#if((SELECT_ACTIVE_FUNCTION == 1) || (SELECT_ACTIVE_FUNCTION == 2))
			//printf("Logistic sigmoid");
			//Logistic sigmoid
			Sigmoid(sum, layer.bias_[i], x, y);//for sigmoid active fun.
#elif(SELECT_ACTIVE_FUNCTION == 3)
			//printf("ActiveRelu");
			//ActiveRelu
			//ActiveRelu(sum, layer.bias_[i], x, y);//for relu active fun.
			ActiveRelu(vec_sum.data(), layer.vec_bias_.at(i), x, y);//for relu active fun.
#endif

			//SetValue(layer.outputmaps_[idx_batch][i], sum, x, y);
			int shift_idx_layer_batch_map = idx_batch * mapNum * x * y;
			int shift_idx_layer_out_map = i * x * y;
			float* p_layer_out_map = layer.vec_output_maps_.data() + shift_idx_layer_batch_map + shift_idx_layer_out_map;
			memcpy(p_layer_out_map, vec_sum.data(), (x * y * sizeof(float)));

		}

		/*for (int i = 0; i < mapNum; i++)
		{
			for (int ii = 0; ii < x; ii++)
			{
				for (int jj = 0; jj < y; jj++)
				{
					std::cout << "FullyConnectedHiddenLayer's active fun. actual output(layer.outputmaps_[" << idx_batch << "][" << i << "][" << ii << "][" << jj << "]): " << layer.outputmaps_[idx_batch][i][ii][jj] << std::endl;
				}
			}
		}*/

	}
}

//for sigmoid & ReLU+Softmax function
void CPUCNN::SetOutLayerOutput(CPUCNNLayer& layer, CPUCNNLayer& lastLayer)
{
	std::cout << "Execute CPUCNN::SetOutLayerOutput()" << std::endl;

	int mapNum = layer.GetOutMapNum();
	int lastMapNum = lastLayer.GetOutMapNum();
	int lastlayer_map_x = lastLayer.GetMapSize().x;
	int lastlayer_map_y = lastLayer.GetMapSize().y;
	int layer_kernel_x = layer.GetKernelSize().x;
	int layer_kernel_y = layer.GetKernelSize().y;
	int x = layer.GetMapSize().x;
	int y = layer.GetMapSize().y;
	vector<float> vec_sum(x * y, 0.0);
	vector<float> vec_sum_now(x * y, 0.0);
	vector<float> vec_sum_expone(batch_size_, 0.0);

	for (int idx_batch = 0; idx_batch < batch_size_; idx_batch++)
	{
#if((SELECT_ACTIVE_FUNCTION == 1) || (SELECT_ACTIVE_FUNCTION == 2))
		//printf("Logistic sigmoid");
		std::cout << "NO.of Batch: " << idx_batch << std::endl;
		for (int i = 0; i < mapNum; i++)
		{
			for (int j = 0; j < lastMapNum; j++)
			{
				float** lastMap;
				lastMap = lastLayer.outputmaps_[idx_batch][j];

				if (j == 0)
				{
					ConvNValid(lastMap, layer.kernel_[j][i], lastlayer_map_x, lastlayer_map_y, layer_kernel_x, layer_kernel_y, sum);

				}
				else {
					ConvNValid(lastMap, layer.kernel_[j][i], lastlayer_map_x, lastlayer_map_y, layer_kernel_x, layer_kernel_y, sumNow);
					CalArrayPlus(sumNow, sum, x, y);

				}
			}

			Sigmoid(sum, layer.bias_[i], x, y);//for sigmoid active fun.
			SetValue(layer.outputmaps_[idx_batch][i], sum, x, y);

		}

		for (int i = 0; i < mapNum; i++)
		{
			for (int ii = 0; ii < x; ii++)
			{
				for (int jj = 0; jj < y; jj++)
				{
					std::cout << "Outputlayer's Sigmoid actual output(layer.outputmaps_[" << idx_batch << "][" << i << "][" << ii << "][" << jj << "]): " << layer.outputmaps_[idx_batch][i][ii][jj] << std::endl;
				}
			}
		}

#elif(SELECT_ACTIVE_FUNCTION == 3)
		//printf("ActiveRelu+softmax");
		//std::cout << "NO.of Batch: " << idx_batch << std::endl;
		for (int i = 0; i < mapNum; i++)
		{
			for (int j = 0; j < lastMapNum; j++)
			{
				int shift_idx_lastlayer_batch_map = idx_batch * lastMapNum * lastlayer_map_x * lastlayer_map_y;
				int shift_idx_lastlayer_out_map = j * lastlayer_map_x * lastlayer_map_y;
				float* p_lastlayer_map = lastLayer.vec_output_maps_.data() + shift_idx_lastlayer_batch_map + shift_idx_lastlayer_out_map;
				//float** lastMap;
				//lastMap = lastLayer.outputmaps_[idx_batch][j];				
				int shift_idx_layer_front_kernel = j * mapNum * layer_kernel_x * layer_kernel_y;
				int shift_idx_layer_out_kernel = i * layer_kernel_x * layer_kernel_y;
				float* p_layer_kernel = layer.vec_kernel_.data() + shift_idx_layer_front_kernel + shift_idx_layer_out_kernel;

				if (j == 0)
				{
					ConvNValid(p_lastlayer_map, p_layer_kernel, lastlayer_map_x, lastlayer_map_y, layer_kernel_x, layer_kernel_y, vec_sum.data());
					//each time we calculate one image of batch and also calculate relu 

				}
				else {
					ConvNValid(p_lastlayer_map, p_layer_kernel, lastlayer_map_x, lastlayer_map_y, layer_kernel_x, layer_kernel_y, vec_sum_now.data());
					CalFCHArrayPlus(vec_sum_now.data(), vec_sum.data(), x, y);// sumNow 

				}
			}

			//CalExpone(sum, layer.bias_[i], x, y);
			CalExpone(vec_sum.data(), layer.vec_bias_.at(i), x, y);


			//SetValue(layer.outputmaps_[idx_batch][i], sum, x, y);
			int shift_idx_layer_batch_map = idx_batch * mapNum * x * y;
			int shift_idx_layer_out_map = i * x * y;
			float* p_layer_out_map = layer.vec_output_maps_.data() + shift_idx_layer_batch_map + shift_idx_layer_out_map;
			memcpy(p_layer_out_map, vec_sum.data(), (x * y * sizeof(float)));

		}

		for (int i = 0; i < mapNum; i++)
		{
			for (int ii = 0; ii < x; ii++)
			{
				for (int jj = 0; jj < y; jj++)
				{
					//sum_Expone[idx_batch] = sum_Expone[idx_batch] + layer.outputmaps_[idx_batch][i][ii][jj];
					int shift_idx_layer_batch_map = idx_batch * mapNum * x * y;
					int shift_idx_layer_out_map = i * x * y;
					int shift_idx_layer_out_map_row = ii * y;
					int idx_layer_out_map = shift_idx_layer_batch_map + shift_idx_layer_out_map + shift_idx_layer_out_map_row + jj;
					vec_sum_expone[idx_batch] += layer.vec_output_maps_.at(idx_layer_out_map);
				}
			}
		}

		for (int i = 0; i < mapNum; i++)
		{
			for (int ii = 0; ii < x; ii++)
			{
				for (int jj = 0; jj < y; jj++)
				{

					//layer.outputmaps_[idx_batch][i][ii][jj] = layer.outputmaps_[idx_batch][i][ii][jj] / sum_Expone[idx_batch];
					
					int shift_idx_layer_batch_map = idx_batch * mapNum * x * y;
					int shift_idx_layer_out_map = i * x * y;
					int shift_idx_layer_out_map_row = ii * y;
					int idx_layer_out_map = shift_idx_layer_batch_map + shift_idx_layer_out_map + shift_idx_layer_out_map_row + jj;
					layer.vec_output_maps_[idx_layer_out_map] = layer.vec_output_maps_[idx_layer_out_map] / vec_sum_expone[idx_batch];
					
					std::cout << "Outputlayer's Softmax actual output(layer.outputmaps_[" << idx_batch << "][" << i << "][" << ii << "][" << jj << "]): " << layer.vec_output_maps_[idx_layer_out_map] << std::endl;
				}
			}
		}
#endif

	}

}


void CPUCNN::SetOutLayerErrors(float**** input_maps, float** target_labels)
{
	VECCPUCNNLayers::iterator iter = vec_layers_.end();
	iter--;
	int mapNum = (*iter).GetOutMapNum();
	float meanError = 0.0, maxError = 0.0;

	FILE* fy;
	fy = fopen("./outputdata/error.txt", "a");

	//if( (err=fopen_s(&fy, "error.txt", "a")) != 0 )
	//	exit(1) ;

	for (int idx_batch = 0; idx_batch < batch_size_; idx_batch++)
	{
		for (int idx_map = 0; idx_map < mapNum; idx_map++)
		{
			//float val_out_map = (*iter).outputmaps_[idx_batch][idx_map][0][0];
			float val_target_label = target_labels[idx_batch][idx_map];
			int shift_idx_layer_batch_map = idx_batch * mapNum * ((*iter).GetMapSize().x) * ((*iter).GetMapSize().y);
			int shift_idx_layer_out_map = idx_map * ((*iter).GetMapSize().x) * ((*iter).GetMapSize().y);
			int shift_idx_layer_out_map_row = 0 * ((*iter).GetMapSize().y);
			int idx_layer_out_map = shift_idx_layer_batch_map + shift_idx_layer_out_map + shift_idx_layer_out_map_row + 0;
			float val_out_map = (*iter).vec_output_maps_.at(idx_layer_out_map);

#if(SELECT_ACTIVE_FUNCTION == 1)
			//printf("quadratic cost function for Logistic sigmoid");
			//quadratic cost function for Logistic sigmoid
			(*iter).SetError(idx_batch, idx_map, 0, 0, DERIV_ACTIVE_SIGMOID(val_out_map) * (val_target_label - val_out_map));
			meanError = abs(val_target_label - val_out_map);
#elif(SELECT_ACTIVE_FUNCTION == 2)
			//printf("Cross entropy cost function for Logistic sigmoid");
			//Cross entropy cost function Logistic sigmoid
			(*iter).SetError(idx_batch, idx_map, 0, 0, (val_target_label - val_out_map));
			meanError = abs(val_target_label - val_out_map);
#elif(SELECT_ACTIVE_FUNCTION == 3)
			//printf("Cross-entropy cost function for ReLU+Softmax");
			//Cross entropy for softmax form
			(*iter).SetError(idx_batch, idx_map, 0, 0, (val_target_label - val_out_map));
			meanError = abs(val_target_label - val_out_map);
#endif

			fprintf(fy, "%f ", meanError);
			// 			meanError += abs(val_target_label-val_out_map);
			// 			if (abs(val_target_label-val_out_map)>maxError)
			// 			{
			// 				maxError = abs(val_target_label-val_out_map);
			// 			}
		}
		fprintf(fy, "\n");
	}
	fprintf(fy, "\n");
	fclose(fy);
	// 	std::cout<<"Mean error of each mini batch: "<<meanError<<std::endl;
	// 	std::cout<<"The max error of one output in mini batch: "<<maxError<<std::endl;
}


void CPUCNN::SetFCHiddenLayerErrors(CPUCNNLayer& Lastlayer, CPUCNNLayer& layer, CPUCNNLayer& nextLayer)//for add FC hiddenlayer
{
	int lastlayer_outmap_num = Lastlayer.GetOutMapNum();
	int layer_outmap_num = layer.GetOutMapNum();
	int layer_outmap_rows = layer.GetMapSize().x;
	int layer_outmap_cols = layer.GetMapSize().y;
	int nextlayer_outmap_num = nextLayer.GetOutMapNum();
	int nextlayer_outmap_rows = nextLayer.GetMapSize().x;
	int nextlayer_outmap_cols = nextLayer.GetMapSize().y;
	
	//float** map;	
	//float** thisError;
	//float** nextError;
	//float** outMatrix, ** kroneckerMatrix;
	//RectSize layer_scale_size;

	float* p_layer_outmap = nullptr;
	float* p_layer_error = nullptr;
	float* p_nextlayer_error = nullptr;
	vector<float> vec_layer_outmatrix;
	vector<float> vec_layer_outkroneckermatrix;
	vec_layer_outmatrix.reserve(layer_outmap_rows * layer_outmap_cols);
	vec_layer_outkroneckermatrix.reserve(layer_outmap_rows * layer_outmap_cols);
	RectSize layer_scale_size = layer.GetScaleSize();

	for (int idx_batch = 0; idx_batch < batch_size_; idx_batch++)
	{
		for (int idx_layer_outmap = 0; idx_layer_outmap < layer_outmap_num; idx_layer_outmap++)
		{
			p_layer_error = layer.GetError(idx_batch, idx_layer_outmap);
			/*for (int ii = 0; ii < layer_outmap_rows; ii++) {
				for (int jj = 0; jj < layer_outmap_cols; jj++) {
					printf("p_layer_error[%d][%d][%d][%d]: %f\n", idx_batch, idx_layer_outmap, ii, jj, p_layer_error[ii][jj]);
				}
			}*/
		}
	}

	for (int idx_batch = 0; idx_batch < batch_size_; idx_batch++)
	{
		for (int idx_nextlayer_outmap = 0; idx_nextlayer_outmap < nextlayer_outmap_num; idx_nextlayer_outmap++)
		{
			p_layer_error = nextLayer.GetError(idx_batch, idx_nextlayer_outmap);
			//for (int ii = 0; ii < nextlayer_outmap_rows; ii++) {
			//	for (int jj = 0; jj < nextlayer_outmap_cols; jj++) {
			//		printf("p_layer_error[%d][%d][%d][%d]: %f\n", idx_batch, idx_nextlayer_outmap, ii, jj, p_layer_error[ii][jj]);
			//	}
			//}
		}
	}
	//printf("================================================================================\n");

	int nextlayer_kernel_rows = nextLayer.GetKernelSize().x;
	int nextlayer_kernel_cols = nextLayer.GetKernelSize().y;
	//float** nextkernel;
	float* p_nextlayer_kernel = nullptr;

	for (int i = 0; i < layer_outmap_num; i++)
	{
		for (int j = 0; j < nextlayer_outmap_num; j++)
		{
			p_nextlayer_kernel = nextLayer.GetKernel(i, j);
			//for (int ii = 0; ii < nextlayer_kernel_rows; ii++)
			//{
			//	for (int jj = 0; jj < nextlayer_kernel_cols; jj++)
			//	{
			//		printf("p_nextlayer_kernel[%d][%d][%d][%d]: %f\n", i, j, ii, jj, p_nextlayer_kernel[ii][jj]);
			//	}
			//}
		}
	}
	//printf("================================================================================\n");

	vector<float> vec_derivative_active_fun;
	vec_derivative_active_fun.reserve(batch_size_ * layer_outmap_num);

	for (int idx_batch = 0; idx_batch < batch_size_; idx_batch++)
	{
		for (int idx_layer_outmap = 0; idx_layer_outmap < layer_outmap_num; idx_layer_outmap++)
		{
			//outMatrix = layer.outputmaps_[idx_batch][idx_layer_outmap];
			int shift_idx_layer_batch_map = idx_batch * layer_outmap_num * layer_outmap_rows * layer_outmap_cols;
			int shift_idx_layer_out_map = idx_layer_outmap * layer_outmap_rows * layer_outmap_cols;
			p_layer_outmap = layer.vec_output_maps_.data() + shift_idx_layer_batch_map + shift_idx_layer_out_map;
			
			layer.SetFCHLayerError(idx_batch, idx_layer_outmap, p_layer_outmap, layer_outmap_rows, layer_outmap_cols);


#if((SELECT_ACTIVE_FUNCTION == 1) || (SELECT_ACTIVE_FUNCTION == 2))
			//printf("derivative of sigmoid");
			//derivative of sigmoid
			matrixDsigmoidFChidden(p_layer_outmap, layer_outmap_rows, layer_outmap_cols, &(derivativeOfActiveFun[idx_batch][idx_layer_outmap]));//for sigmoid active fun. 20171201
#elif(SELECT_ACTIVE_FUNCTION == 3)
			//printf("derivative of ReLu");
			//derivative of ReLu
			//MatrixDreluFChidden(p_layer_outmap, layer_outmap_rows, layer_outmap_cols, &(derivativeOfActiveFun[idx_batch][idx_layer_outmap]));//for relu active fun.
			float* p_derivative_active_fun = vec_derivative_active_fun.data() + (idx_batch * layer_outmap_num + idx_layer_outmap);
			MatrixDreluFChidden(p_layer_outmap, layer_outmap_rows, layer_outmap_cols, p_derivative_active_fun);//for relu active fun.
#endif

		}
	}
	//printf("================================================================================\n");

	for (int idx_batch = 0; idx_batch < batch_size_; idx_batch++)
	{
		for (int idx_layer_outmap = 0; idx_layer_outmap < layer_outmap_num; idx_layer_outmap++)
		{
			p_layer_error = layer.GetError(idx_batch, idx_layer_outmap);
			p_layer_error[0 * layer_outmap_num + 0] = vec_derivative_active_fun.at(idx_batch * layer_outmap_num + idx_layer_outmap);

			//printf("p_layer_error[%d][%d][0][0]: %f\n", idx_batch, idx_layer_outmap, p_layer_error[0][0]);

		}
	}
	//printf("================================================================================\n");

	vector<float> vec_sum_local_gradient((batch_size_ * layer_outmap_num), 0.0);

	for (int idx_batch = 0; idx_batch < batch_size_; idx_batch++)
	{
		for (int idx_layer_outmap = 0; idx_layer_outmap < layer_outmap_num; idx_layer_outmap++)
		{
			for (int idx_nextlayer_outmap = 0; idx_nextlayer_outmap < nextlayer_outmap_num; idx_nextlayer_outmap++)
			{
				p_layer_error = nextLayer.GetError(idx_batch, idx_nextlayer_outmap);
				p_nextlayer_kernel = nextLayer.GetKernel(idx_layer_outmap, idx_nextlayer_outmap);

				//sumOflocalgradient[idx_batch][idx_layer_outmap] += p_layer_error[0][0] * p_nextlayer_kernel[0][0];
				vec_sum_local_gradient[idx_batch * layer_outmap_num + idx_layer_outmap] += p_layer_error[0 * nextlayer_outmap_num + 0] * p_nextlayer_kernel[0 * nextlayer_kernel_cols + 0];

			}

		}
	}

	for (int idx_batch = 0; idx_batch < batch_size_; idx_batch++)
	{
		for (int idx_layer_outmap = 0; idx_layer_outmap < layer_outmap_num; idx_layer_outmap++)
		{
			p_layer_error = layer.GetError(idx_batch, idx_layer_outmap);
			if (0.0 == p_layer_error[0 * nextlayer_outmap_num + 0])
			{
				p_layer_error[0 * nextlayer_outmap_num + 0] = p_layer_error[0 * nextlayer_outmap_num + 0] * vec_sum_local_gradient[idx_batch * layer_outmap_num + idx_layer_outmap];

				p_layer_error[0 * nextlayer_outmap_num + 0] = abs(p_layer_error[0 * nextlayer_outmap_num + 0]);
			}
			else {
				p_layer_error[0 * nextlayer_outmap_num + 0] = p_layer_error[0 * nextlayer_outmap_num + 0] * vec_sum_local_gradient[idx_batch * layer_outmap_num + idx_layer_outmap];
			}

			layer.SetFCHLayerError(idx_batch, idx_layer_outmap, p_layer_error, 0, 0);

		}
	}

}


void CPUCNN::SetHiddenLayerErrors()
{
	VECCPUCNNLayers::iterator iter = vec_layers_.end();
	iter = iter - 2;
	for (iter; iter > vec_layers_.begin(); iter--)
	{
		switch ((*(iter)).GetType())
		{
		case 'C':
			SetConvErrors((*iter), (*(iter + 1)));
			break;
		case 'S':
			SetSampErrors((*iter), (*(iter + 1)));
			break;
		case 'H':
			SetFCHiddenLayerErrors((*(iter - 1)), (*iter), (*(iter + 1)));
			break;
		default:
			break;
		}
	}
}

void CPUCNN::SetSampErrors(CPUCNNLayer& layer, CPUCNNLayer& nextLayer)
{
	int layer_outmap_num = layer.GetOutMapNum();
	int layer_outmap_rows = layer.GetMapSize().x;
	int layer_outmap_cols = layer.GetMapSize().y;
	int nextlayer_outmap_num = nextLayer.GetOutMapNum();
	int nextlayer_outmap_rows = nextLayer.GetMapSize().x;
	int nextlayer_outmap_cols = nextLayer.GetMapSize().y;
	int nextlayer_kernel_rows = nextLayer.GetKernelSize().x;
	int nextlayer_kernel_cols = nextLayer.GetKernelSize().y;
	
	//float** nextError;
	//float** kernel_;
	//float** sum, ** rotMatrix, ** sumNow;

	float* p_nextlayer_error = nullptr;
	float* p_nextlayer_kernel = nullptr;
	vector<float> vec_sum(layer_outmap_rows * layer_outmap_cols, 0.0);
	vector<float> vec_sum_now(layer_outmap_rows * layer_outmap_cols, 0.0);
	vector<float> vec_rot_matrix(nextlayer_kernel_rows * nextlayer_kernel_cols, 0.0);
	vector<float> vec_nextlayer_extend_matrix((nextlayer_outmap_rows+2*(nextlayer_kernel_rows-1)) * (nextlayer_outmap_cols+2*(nextlayer_kernel_cols-1)), 0.0);

	//initialize
	//float** extendMatrix;
	//int m = nextlayer_outmap_rows, n = nextlayer_outmap_cols, km = nextlayer_kernel_rows, kn = nextlayer_kernel_cols;
	//extendMatrix = new float* [m + 2 * (km - 1)];
	//for (int k = 0; k < m + 2 * (km - 1); k++)
	//{
	//	extendMatrix[k] = new float[n + 2 * (kn - 1)];
	//	for (int a = 0; a < n + 2 * (kn - 1); a++)
	//	{
	//		extendMatrix[k][a] = 0.0;
	//	}
	//}
	
	//calculate
	for (int idx_batch = 0; idx_batch < batch_size_; idx_batch++)
	{
		for (int idx_layer_outmap = 0; idx_layer_outmap < layer_outmap_num; idx_layer_outmap++)
		{
			for (int idx_nextlayer_outmap = 0; idx_nextlayer_outmap < nextlayer_outmap_num; idx_nextlayer_outmap++)
			{

				p_nextlayer_error = nextLayer.GetError(idx_batch, idx_nextlayer_outmap);
				p_nextlayer_kernel = nextLayer.GetKernel(idx_layer_outmap, idx_nextlayer_outmap);
				if (idx_nextlayer_outmap == 0)
				{
					Rot180(p_nextlayer_kernel, nextlayer_kernel_rows, nextlayer_kernel_cols, vec_rot_matrix.data());
					ConvNSampFull(p_nextlayer_error, vec_rot_matrix.data(), nextlayer_outmap_rows, nextlayer_outmap_cols, nextlayer_kernel_rows, nextlayer_kernel_cols, vec_sum.data(), vec_nextlayer_extend_matrix.data());

				}
				else
				{
					Rot180(p_nextlayer_kernel, nextlayer_kernel_rows, nextlayer_kernel_cols, vec_rot_matrix.data());
					ConvNSampFull(p_nextlayer_error, vec_rot_matrix.data(), nextlayer_outmap_rows, nextlayer_outmap_cols, nextlayer_kernel_rows, nextlayer_kernel_cols, vec_sum_now.data(), vec_nextlayer_extend_matrix.data());
					CalSampArrayPlus(vec_sum_now.data(), vec_sum.data(), layer_outmap_rows, layer_outmap_cols);

				}

			}
			layer.SetSampLayerError(idx_batch, idx_layer_outmap, vec_sum.data(), layer_outmap_rows, layer_outmap_cols);
		}
	}

}

void CPUCNN::SetConvErrors(CPUCNNLayer& layer, CPUCNNLayer& nextLayer)
{
	int layer_outmap_num = layer.GetOutMapNum();
	int layer_outmap_rows = layer.GetMapSize().x;
	int layer_outmap_cols = layer.GetMapSize().y;
	int nextlayer_outmap_rows = nextLayer.GetMapSize().x;
	int nextlayer_outmap_cols = nextLayer.GetMapSize().y;
	//float** nextError;
	//float** map;
	//float** outMatrix, ** kroneckerMatrix;
	//RectSize scale;

	float* p_nextlayer_error = nullptr;
	float* p_layer_outmap = nullptr;
	vector<float> vec_layer_outmatrix;
	vector<float> vec_layer_outkroneckermatrix;
	vec_layer_outmatrix.reserve(layer_outmap_rows * layer_outmap_cols);
	vec_layer_outkroneckermatrix.reserve(layer_outmap_rows * layer_outmap_cols);
	RectSize layer_scale_size = layer.GetScaleSize();

	for (int idx_batch = 0; idx_batch < batch_size_; idx_batch++)
	{
		for (int idx_layer_outmap = 0; idx_layer_outmap < layer_outmap_num; idx_layer_outmap++)
		{
			layer_scale_size = nextLayer.GetScaleSize();
			p_nextlayer_error = nextLayer.GetError(idx_batch, idx_layer_outmap);
			//p_layer_outmap = layer.outputmaps_[idx_batch][idx_layer_outmap];
			int shift_idx_layer_batch_map = idx_batch * layer_outmap_num * layer_outmap_rows * layer_outmap_cols;
			int shift_idx_layer_out_map = idx_layer_outmap * layer_outmap_rows * layer_outmap_cols;
			p_layer_outmap = layer.vec_output_maps_.data() + shift_idx_layer_batch_map + shift_idx_layer_out_map;


#if((SELECT_ACTIVE_FUNCTION == 1) || (SELECT_ACTIVE_FUNCTION == 2))
			//printf("derivative of sigmoid");
			//derivative of sigmoid
			matrixDsigmoid(p_layer_outmap, layer_outmap_rows, layer_outmap_cols, vec_layer_outmatrix);//for sigmoid active fun.
#elif(SELECT_ACTIVE_FUNCTION == 3)
			//printf("derivative of ReLu");
			//derivative of ReLu
			MatrixDreluConv(p_layer_outmap, layer_outmap_rows, layer_outmap_cols, vec_layer_outmatrix.data());//for relu active fun.
#endif

			CalKronecker(p_nextlayer_error, layer_scale_size, nextlayer_outmap_rows, nextlayer_outmap_cols, vec_layer_outkroneckermatrix.data(), layer_outmap_rows, layer_outmap_cols);
			CalMatrixMultiply(vec_layer_outmatrix.data(), vec_layer_outkroneckermatrix.data(), layer_outmap_rows, layer_outmap_cols);

			layer.SetConvLayerError(idx_batch, idx_layer_outmap, vec_layer_outmatrix.data(), layer_outmap_rows, layer_outmap_cols);

		}
	}

}


void CPUCNN::UpdateParas()
{
	VECCPUCNNLayers::iterator iter = vec_layers_.begin();
	iter++;

	g_idx_itor = 0;//begining at index 0 layer
	char str_file_kernel[1000];// initialized properly
	char str_file_bias[1000];// initialized properly

	for (iter; iter < vec_layers_.end(); iter++)
	{
		g_idx_itor = g_idx_itor + 1;
		sprintf(str_file_kernel, "./data/kernel_weight/kernel_weight_%d_%d", g_idx_itor, (*iter).GetType());
		sprintf(str_file_bias, "./data/bias_/bias_%d_%d", g_idx_itor, (*iter).GetType());
		//printf("%s", str_file_kernel);

		switch ((*iter).GetType())
		{
		case 'C':
			UpdateKernels(*iter, *(iter - 1), str_file_kernel, eta_conv_, alpha_conv_);
			UpdateBias(*iter, str_file_bias, eta_conv_);
			break;
		case 'H':
			UpdateKernels(*iter, *(iter - 1), str_file_kernel, eta_fc_, alpha_fc_);
			UpdateBias(*iter, str_file_bias, eta_fc_);
			break;
		case 'O':
			UpdateKernels(*iter, *(iter - 1), str_file_kernel, eta_fc_, alpha_fc_);
			UpdateBias(*iter, str_file_bias, eta_fc_);
			break;
		default:
			break;
		}
	}
}


void CPUCNN::UpdateBias(CPUCNNLayer& layer, char* str_File_Bias, float eta)
{
	//float**** errors = layer.errors_;
	//float** error;
	int layer_outmap_num = layer.GetOutMapNum();
	int layer_outmap_rows = layer.GetMapSize().x;
	int layer_outmap_cols = layer.GetMapSize().y;
	float* p_layer_errors = layer.vec_errors_.data();
	vector<float> vec_error((layer_outmap_rows* layer_outmap_cols), 0.0);
	float deltaBias = 0.0;

	for (int idx_layer_outmap = 0; idx_layer_outmap < layer_outmap_num; idx_layer_outmap++)
	{

		CalErrorsSum(p_layer_errors, idx_layer_outmap, layer_outmap_num, layer_outmap_rows, layer_outmap_cols, batch_size_, vec_error.data());
		deltaBias = (CalErrorSum(vec_error.data(), layer_outmap_rows, layer_outmap_cols) / batch_size_);
		layer.vec_bias_.at(idx_layer_outmap) = layer.vec_bias_.at(idx_layer_outmap) + (eta * deltaBias);

		/***save bias_***/
		if ((g_iteration_num - 1) == g_idx_iteration_num) {
			char str_file_bias_1[1000];
			sprintf(str_file_bias_1, "%s_%d.txt", str_File_Bias, idx_layer_outmap);
			FILE* fp_bias = fopen(str_file_bias_1, "w");

			fprintf(fp_bias, "%f ", layer.vec_bias_.at(idx_layer_outmap));
			fprintf(fp_bias, "\n");

			fclose(fp_bias);
		}
	}

}

void CPUCNN::UpdateKernels(CPUCNNLayer& layer, CPUCNNLayer& lastLayer, char* str_File_Kernel, float eta, float alpha)
{
	int lastlayer_outmap_num = lastLayer.GetOutMapNum();
	int lastlayer_outmap_rows = lastLayer.GetMapSize().x;
	int lastlayer_outmap_cols = lastLayer.GetMapSize().y;
	int layer_outmap_num = layer.GetOutMapNum();
	int layer_outmap_rows = layer.GetMapSize().x;
	int layer_outmap_cols = layer.GetMapSize().y;
	int layer_kernel_rows = layer.GetKernelSize().x;
	int layer_kernel_cols = layer.GetKernelSize().y;


	vector<float> vec_delta_kernel_1((layer_kernel_rows * layer_kernel_cols), 0.0);
	vector<float> vec_delta_kernel_2((layer_kernel_rows * layer_kernel_cols), 0.0);
	vector<float> vec_delta_now((layer_kernel_rows * layer_kernel_cols), 0.0);
	float* p_layer_error = nullptr;
	float* p_lastlayer_outmap = nullptr;
	float* p_layer_laststep_delta_kernel = nullptr;
	float* p_layer_kernel = nullptr;

	//float** deltakernel1, ** deltakernel2, ** deltaNow;

	for (int idx_layer_outmap = 0; idx_layer_outmap < layer_outmap_num; idx_layer_outmap++)
	{
		for (int idx_lastlayer_outmap = 0; idx_lastlayer_outmap < lastlayer_outmap_num; idx_lastlayer_outmap++)
		{
			for (int idx_batch = 0; idx_batch < batch_size_; idx_batch++)
			{
				//float** error = layer.errors_[idx_batch][idx_layer_outmap];
				p_layer_error = layer.GetError(idx_batch, idx_layer_outmap);
				if (idx_batch == 0) {
					//ConvNValid(lastLayer.outputmaps_[idx_batch][idx_lastlayer_outmap], error, lastlayer_outmap_rows, lastlayer_outmap_cols, layer_outmap_rows, layer_outmap_cols, vec_delta_kernel_1);
					
					int shift_idx_lastlayer_batch_map = idx_batch * lastlayer_outmap_num * lastlayer_outmap_rows * lastlayer_outmap_cols;
					int shift_idx_lastlayer_out_map = idx_lastlayer_outmap * lastlayer_outmap_rows * lastlayer_outmap_cols;
					p_lastlayer_outmap = lastLayer.vec_output_maps_.data() + shift_idx_lastlayer_batch_map + shift_idx_lastlayer_out_map;
					ConvNValid(p_lastlayer_outmap, p_layer_error, lastlayer_outmap_rows, lastlayer_outmap_cols, layer_outmap_rows, layer_outmap_cols, vec_delta_kernel_1.data());
				}
				else {
					//ConvNValid(lastLayer.outputmaps_[idx_batch][idx_lastlayer_outmap], error, lastlayer_outmap_rows, lastlayer_outmap_cols, layer_outmap_rows, layer_outmap_cols, vec_delta_now);
					//CalArrayPlus(vec_delta_now, vec_delta_kernel_1, layer_kernel_rows, layer_kernel_cols);

					int shift_idx_lastlayer_batch_map = idx_batch * lastlayer_outmap_num * lastlayer_outmap_rows * lastlayer_outmap_cols;
					int shift_idx_lastlayer_out_map = idx_lastlayer_outmap * lastlayer_outmap_rows * lastlayer_outmap_cols;
					p_lastlayer_outmap = lastLayer.vec_output_maps_.data() + shift_idx_lastlayer_batch_map + shift_idx_lastlayer_out_map;
					ConvNValid(p_lastlayer_outmap, p_layer_error, lastlayer_outmap_rows, lastlayer_outmap_cols, layer_outmap_rows, layer_outmap_cols, vec_delta_now.data());
					CalConvArrayPlus(vec_delta_now.data(), vec_delta_kernel_1.data(), layer_kernel_rows, layer_kernel_cols);
				}
			}
			//SetValue(vec_delta_kernel_2, layer.laststepdeltakernel_[idx_lastlayer_outmap][idx_layer_outmap], layer.GetKernelSize().x, layer.GetKernelSize().y);//for adding momentum
			int shift_idx_layer_kernel_lastlayer = idx_lastlayer_outmap * layer_outmap_num * layer_kernel_rows * layer_kernel_cols;
			int shift_idx_layer_kernel_layer = idx_layer_outmap * layer_kernel_rows * layer_kernel_cols;
			p_layer_laststep_delta_kernel = layer.vec_laststep_delta_kernel_.data() + shift_idx_layer_kernel_lastlayer + shift_idx_layer_kernel_layer;
			p_layer_kernel = layer.vec_kernel_.data() + shift_idx_layer_kernel_lastlayer + shift_idx_layer_kernel_layer;
			SetKernelValue(vec_delta_kernel_2.data(), p_layer_laststep_delta_kernel, layer_kernel_rows, layer_kernel_cols);
			CalArrayMultiply(vec_delta_kernel_2.data(), alpha, layer_kernel_rows, layer_kernel_cols);//for adding momentum
			CalArrayPlus(vec_delta_kernel_2.data(), p_layer_kernel, layer_kernel_rows, layer_kernel_cols);//for adding momentum
			CalArrayDivide(vec_delta_kernel_1.data(), batch_size_, layer_kernel_rows, layer_kernel_cols);
			CalArrayMultiply(vec_delta_kernel_1.data(), eta, layer_kernel_rows, layer_kernel_cols);
			//SetValue(layer.laststepdeltakernel_[idx_lastlayer_outmap][idx_layer_outmap], vec_delta_kernel_1, layer.GetKernelSize().layer_kernel_rows, layer.GetKernelSize().layer_kernel_cols);//for adding momentum
			SetKernelValue(p_layer_laststep_delta_kernel, vec_delta_kernel_1.data(), layer_kernel_rows, layer_kernel_cols);//for adding momentum
			CalArrayPlus(vec_delta_kernel_1.data(), p_layer_kernel, layer_kernel_rows, layer_kernel_cols);

			/***save kernel_ weight***/
			if ((g_iteration_num - 1) == g_idx_iteration_num) {
				char str_file_kernel_1[1000];
				sprintf(str_file_kernel_1, "%s_%d_%d.txt", str_File_Kernel, idx_lastlayer_outmap, idx_layer_outmap);

				FILE* fp = fopen(str_file_kernel_1, "w");
				int shift_idx_layer_kernel_lastlayer = 0;
				int shift_idx_layer_kernel_layer = 0;
				int idx_layer_kernel = 0;

				for (int mm = 0; mm < layer_kernel_rows; mm++)
				{
					for (int nn = 0; nn < layer_kernel_cols; nn++)
					{
						shift_idx_layer_kernel_lastlayer = idx_lastlayer_outmap * layer_outmap_num * layer_kernel_rows * layer_kernel_cols;
						shift_idx_layer_kernel_layer = idx_layer_outmap * layer_kernel_rows * layer_kernel_cols;
						idx_layer_kernel = shift_idx_layer_kernel_lastlayer + shift_idx_layer_kernel_layer + (mm * layer_kernel_cols + nn);

						fprintf(fp, "%f ", layer.vec_kernel_.at(idx_layer_kernel));
					}

				}
				fprintf(fp, "\n");
				fclose(fp);
			}

		}
	}

}


void CPUCNN::LoadParas()
{
	VECCPUCNNLayers::iterator iter = vec_layers_.begin();
	iter++;

	g_idx_itor = 0;//begining at index 0 layer
	char str_file_kernel[1000];// initialized properly
	char str_file_bias[1000];// initialized properly

	for (iter; iter < vec_layers_.end(); iter++)
	{
		g_idx_itor = g_idx_itor + 1;
		sprintf(str_file_kernel, "./data/kernel_weight/kernel_weight_%d_%d", g_idx_itor, (*iter).GetType());
		sprintf(str_file_bias, "./data/bias_/bias_%d_%d", g_idx_itor, (*iter).GetType());
		//printf("%s", str_file_kernel);

		switch ((*iter).GetType())
		{
		case 'C':
			LoadKernels(*iter, *(iter - 1), str_file_kernel);
			LoadBias(*iter, str_file_bias);
			break;
		case 'H':
			LoadKernels(*iter, *(iter - 1), str_file_kernel);
			LoadBias(*iter, str_file_bias);
			break;
		case 'O':
			LoadKernels(*iter, *(iter - 1), str_file_kernel);
			LoadBias(*iter, str_file_bias);
			break;
		default:
			break;
		}
	}
}

void CPUCNN::LoadBias(CPUCNNLayer& layer, char* str_File_Bias)
{
	int layer_outmap_num = layer.GetOutMapNum();

	for (int idx_layer_outmap = 0; idx_layer_outmap < layer_outmap_num; idx_layer_outmap++)
	{

		float bias;
		/***load bias***/
		char str_file_bias_1[1000];
		sprintf(str_file_bias_1, "%s_%d.txt", str_File_Bias, idx_layer_outmap);
		printf("%s\n", str_file_bias_1);
		FILE* fp_bias = fopen(str_file_bias_1, "r");

		fscanf(fp_bias, "%f ", &bias);
		layer.vec_bias_.at(idx_layer_outmap) = bias;

		printf("bias: %f\n", layer.vec_bias_.at(idx_layer_outmap));
	}

}

void CPUCNN::LoadKernels(CPUCNNLayer& layer, CPUCNNLayer& lastLayer, char* str_File_Kernel)
{
	
	const int lastlayer_outmap_num = lastLayer.GetOutMapNum();
	const int lastlayer_outmap_rows = lastLayer.GetMapSize().x;
	const int lastlayer_outmap_cols = lastLayer.GetMapSize().y;
	const int layer_outmap_num = layer.GetOutMapNum();
	const int layer_outmap_rows = layer.GetMapSize().x;
	const int layer_outmap_cols = layer.GetMapSize().y;
	const int layer_kernel_rows = layer.GetKernelSize().x;
	const int layer_kernel_cols = layer.GetKernelSize().y;

	int shift_idx_layer_kernel_lastlayer = 0;
	int shift_idx_layer_kernel_layer = 0;
	int idx_layer_kernel = 0;

	std::vector<float> vec_kernel((layer_kernel_rows * layer_kernel_cols), 0.0);

	for (int idx_layer_outmap = 0; idx_layer_outmap < layer_outmap_num; idx_layer_outmap++)
	{
		for (int idx_lastlayer_outmap = 0; idx_lastlayer_outmap < lastlayer_outmap_num; idx_lastlayer_outmap++)
		{
			/***load kernel_ weight***/
			char str_file_kernel_1[1000];
			sprintf(str_file_kernel_1, "%s_%d_%d.txt", str_File_Kernel, idx_lastlayer_outmap, idx_layer_outmap);
			printf("%s\n", str_file_kernel_1);
			FILE* fp = fopen(str_file_kernel_1, "r");

			for (int mm = 0; mm < layer_kernel_rows; mm++)
			{
				for (int nn = 0; nn < layer_kernel_cols; nn++)
				{
					shift_idx_layer_kernel_lastlayer = idx_lastlayer_outmap * layer_outmap_num * layer_kernel_rows * layer_kernel_cols;
					shift_idx_layer_kernel_layer = idx_layer_outmap * layer_kernel_rows * layer_kernel_cols;
					idx_layer_kernel = shift_idx_layer_kernel_lastlayer + shift_idx_layer_kernel_layer + (mm * layer_kernel_cols + nn);

					fscanf(fp, "%f ", (vec_kernel.data()+(mm * layer_kernel_cols + nn)));
					layer.vec_kernel_.at(idx_layer_kernel) = vec_kernel.at(mm * layer_kernel_cols + nn);
					printf("kernel_: %f\n", layer.vec_kernel_.at(idx_layer_kernel));
				}

			}

		}
	}
}


void CPUCNN::Inference(float**** test_x, float** test_label, int number)
{
	std::cout << "Start Inference" << std::endl;
	//char _posfilepath[256]="Traing_p_";
	char _posfilepath[256] = "(Training)p_";
	char _negfilepath[256] = "Training_n_";
	char _imgfileextension[16] = ".bmp";
	char _input_posfilename[512];
	char _input_negfilename[512];
	int num_charaters_pos1;
	int num_charaters_neg1;

	int totalfause = 0, fause1 = 0, fause2 = 0, predict, real;
	int Num = number / batch_size_;
	float**** Test;

	Test = new float*** [batch_size_];

	FILE* fy_negfilename;
	fy_negfilename = fopen("./outputdata/error_predict_neg_filename.txt", "w");
	FILE* fy_posfilename;
	fy_posfilename = fopen("./outputdata/error_predict_pos_filename.txt", "w");
	for (int i = 0; i < Num; i++)
	{
		std::cout << "NO.of iteration(testing): " << i << std::endl;


		for (int j = 0; j < batch_size_; j++)
		{
			std::cout << "NO.of batch(testing): " << j << std::endl;
			if (i == 0)
			{
				Test[j] = new float** [g_input_num_channel];
			}
			for (int c = 0; c < g_input_num_channel; c++)
			{

				if (i == 0)
				{
					Test[j][c] = new float* [IMG_H];
				}

				for (int m = 0; m < IMG_H; m++)
				{
					if (i == 0)
					{
						Test[j][c][m] = new float[IMG_W];
					}
					for (int n = 0; n < IMG_W; n++)
					{
						Test[j][c][m][n] = test_x[i * batch_size_ + j][c][m][n];
					}
				}
			}
		}

		Forward(Test);
		VECCPUCNNLayers::iterator iter = vec_layers_.end();
		iter--;
		for (int ii = 0; ii < batch_size_; ii++)
		{
			std::cout << ii << std::endl;

			int layer_outmap_num = (*iter).GetOutMapNum();
			int layer_outmap_rows = (*iter).GetMapSize().x;
			int layer_outmap_cols = (*iter).GetMapSize().y;
			int shift_idx_layer_batch_map = ii * layer_outmap_num * layer_outmap_rows * layer_outmap_cols;		
			float* p_layer_batchmap = (*iter).vec_output_maps_.data() + shift_idx_layer_batch_map;
			predict = FindIndex(p_layer_batchmap, layer_outmap_num, layer_outmap_rows, layer_outmap_cols);
			//predict = findIndex((*iter).outputmaps_[ii]);
			real = FindIndex(test_label[i * batch_size_ + ii]);


			//predict For batch size=2
			if (0 == ii) {
				if (predict != real)
				{
					fause1++;
					num_charaters_neg1 = sprintf(_input_negfilename, "%s%d%s", _negfilepath, i, _imgfileextension);
					//printf("error predict-number of charaters: %d, string: \"%s\"\n", num_charaters_neg1, _input_negfilename);
					fprintf(fy_negfilename, "%s\n", _input_negfilename);

				}
			}
			else if (1 == ii) {
				if (predict != real)
				{
					fause2++;
					//num_charaters_pos1 = sprintf(_input_posfilename, "%s%d%s", _posfilepath, i, _imgfileextension);
					num_charaters_pos1 = sprintf(_input_posfilename, "%s%d", _posfilepath, i);
					//printf("error predict-number of charaters: %d, string: \"%s\"\n", num_charaters_pos1, _input_posfilename);
					fprintf(fy_posfilename, "%s\n", _input_posfilename);
				}
			}


			/*predict for batchsize = 10
			if(9 > ii){
				if(predict != real)
				{
					fause1++;
					num_charaters_neg1 = sprintf(_input_negfilename, "%s%d%s", _negfilepath, ((9*i)+ii), _imgfileextension);
					//printf("error predict-number of charaters: %d, string: \"%s\"\n", num_charaters_neg1, _input_negfilename);
					fprintf(fy_negfilename, "%s\n", _input_negfilename);

				}
			}else if(9 == ii){
				if(predict != real)
				{
					fause2++;
					//num_charaters_pos1 = sprintf(_input_posfilename, "%s%d%s", _posfilepath, i, _imgfileextension);
					num_charaters_pos1 = sprintf(_input_posfilename, "%s%d", _posfilepath, i);
					//printf("error predict-number of charaters: %d, string: \"%s\"\n", num_charaters_pos1, _input_posfilename);
					fprintf(fy_posfilename, "%s\n", _input_posfilename);
				}
			}
			*/
		}
	}

	totalfause = fause1 + fause2;

	std::cout << "+++++++Finish Inference+++++++" << std::endl;
	std::cout << "Error predict number of neg: " << fause1 << std::endl;
	std::cout << "Error rate of neg: " << 1.0 * fause1 / number << std::endl;
	std::cout << "Error predict number of pos: " << fause2 << std::endl;
	std::cout << "Error rate of pos: " << 1.0 * fause2 / number << std::endl;
	std::cout << "Error predict total number: " << totalfause << std::endl;
	std::cout << "Total error rate: " << 1.0 * totalfause / number << std::endl << std::endl;

	FILE* fy;
	fy = fopen("./outputdata/fausePrun.txt", "a");
	/*
	if( (err=fopen_s(&fy, "fausePrun.txt", "a")) != 0 )
		exit(1) ;
	*/
	g_idx_epoch++;
	fprintf(fy, "epoch: %4d\n", g_idx_epoch);
	fprintf(fy, "neg: %4d %8f\n", fause1, 1.0 * fause1 / number);
	fprintf(fy, "pos: %4d %8f\n", fause2, 1.0 * fause2 / number);
	fprintf(fy, "total: %4d %8f\n\n", totalfause, 1.0 * totalfause / number);
	fclose(fy);
	fclose(fy_posfilename);
	fclose(fy_negfilename);

	for (int i = 0; i < batch_size_; i++)
	{
		for (int c = 0; c < g_input_num_channel; c++)
		{
			for (int j = 0; j < IMG_H; j++)
			{
				delete[]Test[i][c][j];
			}
			delete[]Test[i][c];
		}
		delete[]Test[i];
	}
	delete[]Test;
}
