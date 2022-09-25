/******************************************************************************
Date:  2022/09
Author: CHU-MIN, NIEN
Description:
******************************************************************************/

#include "CNNCls.h"

using namespace std;

// Utility
double cpu_time()
{
	return clock() / CLOCKS_PER_SEC;
}

void randomMatrix(int x, int y, double** matrix)
{
	// construct a trivial random generator engine from a time-based seed:
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<double> distribution(0, 0.1);
	for (int i = 0; i < x; i++)
	{

		for (int j = 0; j < y; j++)
		{
			matrix[i][j] = distribution(generator);
		}
	}
}

void convnValid(double** matrix, double** kernel, int m, int n, int km, int kn, double** outmatrix)
{


	// row
	int kms = m - km + 1;
	// the number of column of convolution
	int kns = n - kn + 1;

	for (int i = 0; i < kms; i++)
	{

		for (int j = 0; j < kns; j++)
		{
			double sum = 0.0;
			for (int ki = 0; ki < km; ki++)
			{
				for (int kj = 0; kj < kn; kj++)
				{
					sum += matrix[i + ki][j + kj] * kernel[ki][kj];
				}
			}
			outmatrix[i][j] = sum;
		}
	}
}

void Relu(double** matrix, double bias, int m, int n)
{


	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			double x1 = RELU(matrix[i][j] + bias);


			if (x1 > 0.0) {
				matrix[i][j] = x1;
			}
			else if (0.0 == x1 || x1 < 0.0) {

				matrix[i][j] = 0;//0.01

			}
			else {
				exit(0);
			}

		}
	}
}

void Sigmoid(double** matrix, double bias, int m, int n)
{


	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			matrix[i][j] = SIGMOID(matrix[i][j] + bias);
		}
	}
}

void Expone(double** matrix, double bias, int m, int n)
{


	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cout << "Outputlayer's actual ouput(matrix[" << i << "][" << j << "] + bias): " << matrix[i][j] + bias << endl;
			matrix[i][j] = exp(matrix[i][j] + bias);
			cout << "Outputlayer's expone actual ouput: " << matrix[i][j] << endl;
		}
	}
}

void ArrayPlus(double** x, double** y, int m, int n)
{

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			y[i][j] = x[i][j] + y[i][j];
		}
	}
}

void scaleMatrix(double** matrix, RectSize scale, int m, int n, double** outMatrix)
{

	int sm = m / scale.x;
	int sn = n / scale.y;
	if (sm * scale.x != m || sn * scale.y != n)
	{
		cout << "scale can not divide by matrix";
	}
	int s = scale.x * scale.y;
	for (int i = 0; i < sm; i++) {
		for (int j = 0; j < sn; j++) {
			double sum = 0.0;
			for (int si = i * scale.x; si < (i + 1) * scale.x; si++) {
				for (int sj = j * scale.y; sj < (j + 1) * scale.y; sj++) {
					sum += matrix[si][sj];
				}
			}
			outMatrix[i][j] = sum / s;
		}
	}
}

void rot180(double** matrix, int m, int n, double** M)
{

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			M[i][j] = matrix[i][n - 1 - j];
		}
	}

	for (int j = 0; j < n; j++)
	{
		for (int i = 0; i < m / 2; i++)
		{
			double tmp = M[i][j];
			M[i][j] = M[m - 1 - i][j];
			M[m - 1 - i][j] = tmp;
		}
	}
}

void convnFull(double** matrix, double** kernel, int m, int n, int km, int kn, double** outmatrix, double** extendMatrix)
{

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++)
			extendMatrix[i + km - 1][j + kn - 1] = matrix[i][j];
	}

	convnValid(extendMatrix, kernel, m + 2 * (km - 1), n + 2 * (kn - 1), km, kn, outmatrix);

}

void matrixDrelu(double** matrix, int m, int n, double** M)
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
void matrixDreluFChidden(double** matrix, int m, int n, double* M)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{

			if (matrix[i][j] > 0.0)
			{
				*M = 1.0;
			}
			else if (0.0 == matrix[i][j] || matrix[i][j] < 0.0) {
				*M = 0.0;
			}

		}
	}
}

void matrixDsigmoid(double** matrix, int m, int n, double** M)
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
void matrixDsigmoidFChidden(double** matrix, int m, int n, double* M)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			*M = matrix[i][j] * (1 - matrix[i][j]);
		}
	}
}

void kronecker(double** matrix, RectSize scale, int m, int n, double** OutMatrix)
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

void matrixMultiply(double** matrix1, double** matrix2, int m, int n)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			matrix1[i][j] = matrix1[i][j] * matrix2[i][j];
		}
	}
}

void sum(double**** errors, int j, int m, int n, int batchSize, double** M)
{
	for (int mi = 0; mi < m; mi++) {
		for (int nj = 0; nj < n; nj++) {
			double sum = 0;
			for (int i = 0; i < batchSize; i++) {
				sum += errors[i][j][mi][nj];
			}
			M[mi][nj] = sum;
		}
	}
}

double sum(double** error, int m, int n)
{
	double sum = 0.0;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			sum += error[i][j];
		}
	}
	return sum;
}

void ArrayDivide(double** M, int batchSize, int m, int n)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			M[i][j] = M[i][j] / batchSize;
		}
	}
}

void ArrayMultiply(double** matrix, double val, int m, int n)
{

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			matrix[i][j] = matrix[i][j] * val;
		}
	}
}

int findIndex(double*** p)
{

	FILE* fy;
	fy = fopen("./outputdata/outputmaps.txt", "a");
	/*
	if( (err=fopen_s(&fy, "outputmaps.txt", "a")) != 0 )
		exit(1) ;
	*/
	int index = 0;
	double v;
	double Max = p[0][0][0];
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

int findIndex(double* p)
{
	int index = 0;
	double Max = p[0];
	for (int i = 1; i < 2; i++)
	{
		double v = p[i];
		if (p[i] > Max)
		{
			Max = p[i];
			index = i;
		}
	}
	return index;
}

void setValue(double** maps, double** sum, int m, int n)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{

			maps[i][j] = sum[i][j];
		}
	}
}


// Layer
Layer Layer::buildInputLayer(int InputLayerOutMapNum, RectSize mapsize)
{
	Layer layer;
	layer.type = 'I';
	layer.outMapNum = InputLayerOutMapNum;
	layer.mapSize = mapsize;
	return layer;
}

Layer Layer::buildConvLayer(int outMapNum, RectSize kernelSize)
{
	Layer layer;
	layer.type = 'C';
	layer.outMapNum = outMapNum;
	layer.kernelSize = kernelSize;
	return layer;
}

Layer Layer::buildSampLayer(RectSize scaleSize)
{
	Layer layer;
	layer.type = 'S';
	layer.scaleSize = scaleSize;
	return layer;
}

Layer Layer::buildFullyConnectedHiddenLayer(int classNum)
{
	Layer layer;
	layer.classNum = classNum;
	layer.type = 'H';
	layer.mapSize = RectSize(1, 1);
	layer.outMapNum = classNum;
	return layer;

}

Layer Layer::buildOutputLayer(int classNum)
{
	Layer layer;
	layer.classNum = classNum;
	layer.type = 'O';
	layer.mapSize = RectSize(1, 1);
	layer.outMapNum = classNum;
	return layer;

}


void Layer::initKernel(int frontMapNum)
{
	kernel = new double*** [frontMapNum];
	for (int i = 0; i < frontMapNum; i++)
	{
		kernel[i] = new double** [outMapNum];
		for (int j = 0; j < outMapNum; j++)
		{
			kernel[i][j] = new double* [kernelSize.x];
			for (int ii = 0; ii < kernelSize.x; ii++)
			{
				kernel[i][j][ii] = new double[kernelSize.y];
			}
			randomMatrix(kernelSize.x, kernelSize.y, kernel[i][j]);
		}
	}

}

//for adding momentum
void Layer::initLastStepDeltaKernel(int frontMapNum)
{
	laststepdeltakernel = new double*** [frontMapNum];
	for (int i = 0; i < frontMapNum; i++)
	{
		laststepdeltakernel[i] = new double** [outMapNum];
		for (int j = 0; j < outMapNum; j++)
		{
			laststepdeltakernel[i][j] = new double* [kernelSize.x];
			for (int ii = 0; ii < kernelSize.x; ii++)
			{
				laststepdeltakernel[i][j][ii] = new double[kernelSize.y];
			}
			for (int iii = 0; iii < kernelSize.x; iii++)
			{
				for (int jjj = 0; jjj < kernelSize.y; jjj++)
				{

					laststepdeltakernel[i][j][iii][jjj] = 0.0;
				}
			}
		}
	}

}

void Layer::initOutputKernel(int frontMapNum, RectSize s)
{
	kernelSize = s;
	kernel = new double*** [frontMapNum];
	for (int i = 0; i < frontMapNum; i++)
	{
		kernel[i] = new double** [outMapNum];
		for (int j = 0; j < outMapNum; j++)
		{
			kernel[i][j] = new double* [kernelSize.x];
			for (int ii = 0; ii < kernelSize.x; ii++)
			{
				kernel[i][j][ii] = new double[kernelSize.y];
			}
			randomMatrix(kernelSize.x, kernelSize.y, kernel[i][j]);
		}
	}

}

//for adding momentum
void Layer::initOutputLastStepDeltaKernel(int frontMapNum, RectSize s)
{
	kernelSize = s;
	laststepdeltakernel = new double*** [frontMapNum];
	for (int i = 0; i < frontMapNum; i++)
	{
		laststepdeltakernel[i] = new double** [outMapNum];
		for (int j = 0; j < outMapNum; j++)
		{
			laststepdeltakernel[i][j] = new double* [kernelSize.x];
			for (int ii = 0; ii < kernelSize.x; ii++)
			{
				laststepdeltakernel[i][j][ii] = new double[kernelSize.y];
			}
			for (int iii = 0; iii < kernelSize.x; iii++)
			{
				for (int jjj = 0; jjj < kernelSize.y; jjj++)
				{

					laststepdeltakernel[i][j][iii][jjj] = 0.0;
				}
			}
		}
	}

}

void Layer::initErros(int batchSize)
{
	errors = new double*** [batchSize];
	for (int i = 0; i < batchSize; i++)
	{
		errors[i] = new double** [outMapNum];
		for (int m = 0; m < outMapNum; m++)
		{
			errors[i][m] = new double* [mapSize.x];
			for (int n = 0; n < mapSize.x; n++)
			{
				errors[i][m][n] = new double[mapSize.y];
			}
		}
	}
}

void Layer::setError(int num, int mapNo, int mapX, int mapY, double v)
{

	errors[num][mapNo][mapX][mapY] = v;

}

void Layer::setError(int numBatch, int mapNo, double** matrix, int m, int n)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			errors[numBatch][mapNo][i][j] = matrix[i][j];

		}
	}
}


// CNN
#define SELECT_ACTIVE_FUNCTION 3 
//'1' : active function is "sigmoid" , cost function is "quadratic" for "sigmoid" form
//'2' : active function is "sigmoid" , cost function is "Cross entropy" for "sigmoid" form
//'3' : active function is "ReLU" , cost function is "Cross entropy" for "softmax" form

#define DRELU(S) 1 // derivative of the relu as a function of the relu's output
#define DSIGMOID(S) (S*(1-S)) // derivative of the sigmoid as a function of the sigmoid's output

int NumOfChannel = 3;//image's channel

int runi = 0;//index of epoch
int no_iter = 0;//index of iterator
int no_iter_initbias = 0;//index of iterator for initialize bias
int iOfiterationsNum = 0;//index of iteration
int iterationsNum = 0;//number of iterationsNum

void CNN::train(double**** train_x, double** train_label, int NumOfImage)
{
	cout << "Start train" << endl;

	iterationsNum = NumOfImage / batchSize;
	if (NumOfImage % batchSize != 0)
	{
		cout << "Please reset batchSize!" << endl;
	}
	double**** Train;
	double** TrainLabel;

	Train = new double*** [batchSize];
	TrainLabel = new double* [batchSize];

	for (iOfiterationsNum = 0; iOfiterationsNum < iterationsNum; iOfiterationsNum++)
	{

		cout << "NO.of iteration(training): " << iOfiterationsNum << endl;
		int ii = iOfiterationsNum % (NumOfImage / batchSize);
		for (int j = 0; j < batchSize; j++)
		{

			cout << "NO.of batch(training): " << j << endl;

			if (iOfiterationsNum == 0)
			{
				Train[j] = new double** [NumOfChannel];
			}
			for (int c = 0; c < NumOfChannel; c++)
			{

				if (iOfiterationsNum == 0)
				{
					Train[j][c] = new double* [IMG_H];
				}
				for (int m = 0; m < IMG_H; m++)
				{
					if (iOfiterationsNum == 0)
					{
						Train[j][c][m] = new double[IMG_W];
					}
					for (int n = 0; n < IMG_W; n++)
					{
						Train[j][c][m][n] = train_x[ii * batchSize + j][c][m][n];

					}
				}
				if (iOfiterationsNum == 0)
				{
					TrainLabel[j] = new double[2];
				}
				for (int l = 0; l < 2; l++)
				{
					TrainLabel[j][l] = train_label[ii * batchSize + j][l];
				}
			}
		}


		forward(Train);
		backPropagation(Train, TrainLabel);
		updateParas();


	}
	cout << "Finish train" << endl;

	for (int i = 0; i < batchSize; i++)
	{
		delete[]TrainLabel[i];
		for (int c = 0; c < NumOfChannel; c++)
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

void CNN::setup(int batchSize)
{
	layers::iterator iter = m_layers.begin();

	(*iter).initOutputmaps(batchSize);
	iter++;
	for (iter; iter < m_layers.end(); iter++)
	{
		no_iter_initbias = no_iter_initbias + 1;

		int frontMapNum = (*(iter - 1)).getOutMapNum();

		switch ((*iter).getType())
		{
		case 'I':
			break;
		case 'C':
			// set map RectSize
			(*iter).setMapSize((*(iter - 1)).getMapSize().substract((*iter).getKernelSize(), 1));
			// initial convolution kernel, quantities: frontMapNum*outMapNum
			(*iter).initKernel(frontMapNum);
			(*iter).initLastStepDeltaKernel(frontMapNum);//for adding momentum
			//each map has one bias, so frontMapNum is not necessary
			(*iter).initBias(frontMapNum, no_iter_initbias);
			(*iter).initErros(batchSize);
			// each layer should initialize output map
			(*iter).initOutputmaps(batchSize);
			break;
		case 'S':
			(*iter).setOutMapNum((frontMapNum));
			(*iter).setMapSize((*(iter - 1)).getMapSize().divide((*iter).getScaleSize()));
			(*iter).initErros(batchSize);
			(*iter).initOutputmaps(batchSize);
			break;
		case 'H':
			(*iter).initOutputKernel(frontMapNum, (*(iter - 1)).getMapSize());
			(*iter).initOutputLastStepDeltaKernel(frontMapNum, (*(iter - 1)).getMapSize());//for adding momentum			
			(*iter).initBias(frontMapNum, no_iter_initbias);
			(*iter).initErros(batchSize);
			(*iter).initOutputmaps(batchSize);
			break;
		case 'O':
			(*iter).initOutputKernel(frontMapNum, (*(iter - 1)).getMapSize());
			(*iter).initOutputLastStepDeltaKernel(frontMapNum, (*(iter - 1)).getMapSize());//for adding momentum
			(*iter).initBias(frontMapNum, no_iter_initbias);
			(*iter).initErros(batchSize);
			(*iter).initOutputmaps(batchSize);
			break;
		default:
			break;
		}
	}
}


void CNN::backPropagation(double**** x, double** y)
{
	setOutLayerErrors(x, y);
	setHiddenLayerErrors();
}

void CNN::forward(double**** x)
{
	setInLayerOutput(x);
	layers::iterator iter = m_layers.begin();
	iter++;
	for (iter; iter < m_layers.end(); iter++)
	{
		switch ((*iter).getType())
		{
		case 'C':
			setConvOutput((*iter), (*(iter - 1)));
			break;
		case 'S':
			setSampOutput((*iter), (*(iter - 1)));
			break;
		case 'H':
			setFullyConnectedHiddenLayerOutput((*iter), (*(iter - 1)));
			break;
		case 'O':
			setOutLayerOutput((*iter), (*(iter - 1)));
			break;
		default:
			break;
		}

	}
}

void CNN::setInLayerOutput(double**** x)
{
	layers::iterator iter = m_layers.begin();

	RectSize mapSize = (*iter).getMapSize();
	if (IMG_H != mapSize.x)
	{
		cout << "IMG_H != mapSize.x" << endl;
	}
	for (int i = 0; i < batchSize; i++)
	{

		for (int c = 0; c < NumOfChannel; c++)
		{
			setValue((*iter).outputmaps[i][c], x[i][c], IMG_H, IMG_W);
		}
	}
}
// for change the value in m_Layers
void CNN::setConvOutput(Layer& layer, Layer& lastLayer)
{
	int mapNum = layer.getOutMapNum();
	int lastMapNum = lastLayer.getOutMapNum();
	int last_x = lastLayer.getMapSize().x;
	int last_y = lastLayer.getMapSize().y;
	int kernel_x = layer.getKernelSize().x;
	int kernel_y = layer.getKernelSize().y;
	int x = layer.getMapSize().x;
	int y = layer.getMapSize().y;
	double** sum, ** sumNow;
	sum = new double* [x];
	sumNow = new double* [x];
	for (int nn = 0; nn < x; nn++)
	{
		sum[nn] = new double[y];
		sumNow[nn] = new double[y];
	}

	//init sum and sumNow
	for (int i = 0; i < x; i++)
	{
		for (int j = 0; j < y; j++)
		{
			sum[i][j] = 0;
			sumNow[i][j] = 0;
		}
	}

	int numBatch;
	for (numBatch = 0; numBatch < batchSize; numBatch++)
	{
		for (int i = 0; i < mapNum; i++)
		{

			for (int j = 0; j < lastMapNum; j++)
			{
				double** lastMap;
				lastMap = lastLayer.outputmaps[numBatch][j];

				if (j == 0)
				{
					convnValid(lastMap, layer.kernel[j][i], last_x, last_y, kernel_x, kernel_y, sum);
					//each time we calculate one image of batch and also calculate relu 

				}
				else {
					convnValid(lastMap, layer.kernel[j][i], last_x, last_y, kernel_x, kernel_y, sumNow);
					ArrayPlus(sumNow, sum, x, y);// sumNow 

				}
			}

#if((SELECT_ACTIVE_FUNCTION == 1) || (SELECT_ACTIVE_FUNCTION == 2))
			//printf("Logistic sigmoid");
			//Logistic sigmoid
			Sigmoid(sum, layer.bias[i], x, y);//for sigmoid active fun.
#elif(SELECT_ACTIVE_FUNCTION == 3)
			//printf("Relu");
			//Relu
			Relu(sum, layer.bias[i], x, y);//for relu active fun.
#endif

			setValue(layer.outputmaps[numBatch][i], sum, x, y);

		}
	}
	for (int i = 0; i < x; i++)
	{
		delete[]sum[i];
		delete[]sumNow[i];
	}
	delete[]sum;
	delete[]sumNow;
}

void CNN::setSampOutput(Layer& layer, Layer& lastLayer)
{
	int lastMapNum = lastLayer.getOutMapNum();
	int last_x = lastLayer.getMapSize().x;
	int last_y = lastLayer.getMapSize().y;
	int x = layer.getMapSize().x;
	int y = layer.getMapSize().y;
	double** sampMatrix, ** lastMap;
	RectSize scaleSize;
	sampMatrix = new double* [layer.getMapSize().x];
	for (int j = 0; j < layer.getMapSize().x; j++)
	{
		sampMatrix[j] = new double[layer.getMapSize().y];
	}

	for (int num = 0; num < batchSize; num++)
	{
		for (int i = 0; i < lastMapNum; i++)
		{
			lastMap = lastLayer.outputmaps[num][i];
			scaleSize = layer.getScaleSize();
			scaleMatrix(lastMap, scaleSize, last_x, last_y, sampMatrix);

			setValue(layer.outputmaps[num][i], sampMatrix, x, y);

		}
	}
	for (int i = 0; i < x; i++)
	{
		delete[]sampMatrix[i];
	}
	delete[]sampMatrix;
}

void CNN::setFullyConnectedHiddenLayerOutput(Layer& layer, Layer& lastLayer)
{
	int mapNum = layer.getOutMapNum();
	int lastMapNum = lastLayer.getOutMapNum();
	int last_x = lastLayer.getMapSize().x;
	int last_y = lastLayer.getMapSize().y;
	int kernel_x = layer.getKernelSize().x;
	int kernel_y = layer.getKernelSize().y;
	int x = layer.getMapSize().x;
	int y = layer.getMapSize().y;
	int numBatch;

	double** sum, ** sumNow;
	sum = new double* [x];
	sumNow = new double* [x];

	for (int nn = 0; nn < x; nn++)
	{
		sum[nn] = new double[y];
		sumNow[nn] = new double[y];
	}

	//init sum and sumNow
	for (int i = 0; i < x; i++)
	{
		for (int j = 0; j < y; j++)
		{
			sum[i][j] = 0;
			sumNow[i][j] = 0;
		}
	}

	for (numBatch = 0; numBatch < batchSize; numBatch++)
	{

		for (int i = 0; i < mapNum; i++)
		{

			for (int j = 0; j < lastMapNum; j++)
			{
				double** lastMap;
				lastMap = lastLayer.outputmaps[numBatch][j];

				if (j == 0)
				{
					convnValid(lastMap, layer.kernel[j][i], last_x, last_y, kernel_x, kernel_y, sum);

				}
				else {
					convnValid(lastMap, layer.kernel[j][i], last_x, last_y, kernel_x, kernel_y, sumNow);
					ArrayPlus(sumNow, sum, x, y);// sumNow 

				}
			}

#if((SELECT_ACTIVE_FUNCTION == 1) || (SELECT_ACTIVE_FUNCTION == 2))
			//printf("Logistic sigmoid");
			//Logistic sigmoid
			Sigmoid(sum, layer.bias[i], x, y);//for sigmoid active fun.
#elif(SELECT_ACTIVE_FUNCTION == 3)
			//printf("Relu");
			//Relu
			Relu(sum, layer.bias[i], x, y);//for relu active fun.
#endif

			setValue(layer.outputmaps[numBatch][i], sum, x, y);


		}

		for (int i = 0; i < mapNum; i++)
		{
			for (int ii = 0; ii < x; ii++)
			{
				for (int jj = 0; jj < y; jj++)
				{
					cout << "FullyConnectedHiddenLayer's active fun. actual output(layer.outputmaps[" << numBatch << "][" << i << "][" << ii << "][" << jj << "]): " << layer.outputmaps[numBatch][i][ii][jj] << endl;
				}
			}
		}

	}

	for (int i = 0; i < x; i++)
	{
		delete[]sum[i];
		delete[]sumNow[i];
	}
	delete[]sum;
	delete[]sumNow;
}

//for sigmoid & ReLU+Softmax function
void CNN::setOutLayerOutput(Layer& layer, Layer& lastLayer)
{
	int mapNum = layer.getOutMapNum();
	int lastMapNum = lastLayer.getOutMapNum();
	int last_x = lastLayer.getMapSize().x;
	int last_y = lastLayer.getMapSize().y;
	int kernel_x = layer.getKernelSize().x;
	int kernel_y = layer.getKernelSize().y;
	int x = layer.getMapSize().x;
	int y = layer.getMapSize().y;
	double** sum, ** sumNow;
	double* sum_Expone;
	sum = new double* [x];
	sumNow = new double* [x];
	sum_Expone = new double[batchSize];



	for (int nn = 0; nn < x; nn++)
	{
		sum[nn] = new double[y];
		sumNow[nn] = new double[y];
	}

	//init sum and sumNow
	for (int i = 0; i < x; i++)
	{
		for (int j = 0; j < y; j++)
		{
			sum[i][j] = 0;
			sumNow[i][j] = 0;
		}
	}

	//init sum_Expone
	int numBatch;

	for (numBatch = 0; numBatch < batchSize; numBatch++)
	{
		sum_Expone[numBatch] = 0;
	}

	for (numBatch = 0; numBatch < batchSize; numBatch++)
	{
#if((SELECT_ACTIVE_FUNCTION == 1) || (SELECT_ACTIVE_FUNCTION == 2))
		//printf("Logistic sigmoid");
		cout << "NO.of Batch: " << numBatch << endl;
		for (int i = 0; i < mapNum; i++)
		{
			for (int j = 0; j < lastMapNum; j++)
			{
				double** lastMap;
				lastMap = lastLayer.outputmaps[numBatch][j];

				if (j == 0)
				{
					convnValid(lastMap, layer.kernel[j][i], last_x, last_y, kernel_x, kernel_y, sum);

				}
				else {
					convnValid(lastMap, layer.kernel[j][i], last_x, last_y, kernel_x, kernel_y, sumNow);
					ArrayPlus(sumNow, sum, x, y);

				}
			}

			Sigmoid(sum, layer.bias[i], x, y);//for sigmoid active fun.
			setValue(layer.outputmaps[numBatch][i], sum, x, y);

		}

		for (int i = 0; i < mapNum; i++)
		{
			for (int ii = 0; ii < x; ii++)
			{
				for (int jj = 0; jj < y; jj++)
				{
					cout << "Outputlayer's Sigmoid actual output(layer.outputmaps[" << numBatch << "][" << i << "][" << ii << "][" << jj << "]): " << layer.outputmaps[numBatch][i][ii][jj] << endl;
				}
			}
		}

#elif(SELECT_ACTIVE_FUNCTION == 3)
		//printf("Relu+softmax");
		cout << "NO.of Batch: " << numBatch << endl;
		for (int i = 0; i < mapNum; i++)
		{

			for (int j = 0; j < lastMapNum; j++)
			{
				double** lastMap;
				lastMap = lastLayer.outputmaps[numBatch][j];

				if (j == 0)
				{
					convnValid(lastMap, layer.kernel[j][i], last_x, last_y, kernel_x, kernel_y, sum);

				}
				else {
					convnValid(lastMap, layer.kernel[j][i], last_x, last_y, kernel_x, kernel_y, sumNow);
					ArrayPlus(sumNow, sum, x, y);

				}
			}

			Expone(sum, layer.bias[i], x, y);
			setValue(layer.outputmaps[numBatch][i], sum, x, y);

		}
		for (int i = 0; i < mapNum; i++)
		{
			for (int ii = 0; ii < x; ii++)
			{

				for (int jj = 0; jj < y; jj++)
				{
					sum_Expone[numBatch] = sum_Expone[numBatch] + layer.outputmaps[numBatch][i][ii][jj];
				}
			}
		}

		for (int i = 0; i < mapNum; i++)
		{
			for (int ii = 0; ii < x; ii++)
			{
				for (int jj = 0; jj < y; jj++)
				{

					layer.outputmaps[numBatch][i][ii][jj] = layer.outputmaps[numBatch][i][ii][jj] / sum_Expone[numBatch];
					cout << "Outputlayer's Softmax actual output(layer.outputmaps[" << numBatch << "][" << i << "][" << ii << "][" << jj << "]): " << layer.outputmaps[numBatch][i][ii][jj] << endl;
				}
			}
		}
#endif

	}


	for (int i = 0; i < x; i++)
	{
		delete[]sum[i];
		delete[]sumNow[i];
	}
	delete[]sum;
	delete[]sumNow;
	delete[]sum_Expone;
}


void CNN::setOutLayerErrors(double**** x, double** y)
{
	layers::iterator iter = m_layers.end();
	iter--;
	int mapNum = (*iter).getOutMapNum();
	double meanError = 0.0, maxError = 0.0;

	FILE* fy;
	fy = fopen("./outputdata/error.txt", "a");

	//if( (err=fopen_s(&fy, "error.txt", "a")) != 0 )
	//	exit(1) ;

	for (int numBatch = 0; numBatch < batchSize; numBatch++)
	{
		for (int m = 0; m < mapNum; m++)
		{
			double outmaps = (*iter).outputmaps[numBatch][m][0][0];
			double target = y[numBatch][m];

#if(SELECT_ACTIVE_FUNCTION == 1)
			//printf("quadratic cost function for Logistic sigmoid");
			//quadratic cost function for Logistic sigmoid
			(*iter).setError(numBatch, m, 0, 0, DSIGMOID(outmaps) * (target - outmaps));
			meanError = abs(target - outmaps);
#elif(SELECT_ACTIVE_FUNCTION == 2)
			//printf("Cross entropy cost function for Logistic sigmoid");
			//Cross entropy cost function Logistic sigmoid
			(*iter).setError(numBatch, m, 0, 0, (target - outmaps));
			meanError = abs(target - outmaps);
#elif(SELECT_ACTIVE_FUNCTION == 3)
			//printf("Cross-entropy cost function for ReLU+Softmax");
			//Cross entropy for softmax form
			(*iter).setError(numBatch, m, 0, 0, (target - outmaps));
			meanError = abs(target - outmaps);
#endif

			fprintf(fy, "%f ", meanError);
			// 			meanError += abs(target-outmaps);
			// 			if (abs(target-outmaps)>maxError)
			// 			{
			// 				maxError = abs(target-outmaps);
			// 			}
		}
		fprintf(fy, "\n");
	}
	fprintf(fy, "\n");
	fclose(fy);
	// 	cout<<"Mean error of each mini batch: "<<meanError<<endl;
	// 	cout<<"The max error of one output in mini batch: "<<maxError<<endl;
}


void CNN::setFCHiddenLayerErrors(Layer& Lastlayer, Layer& layer, Layer& nextLayer)//for add FC hiddenlayer
{
	int mapNum = layer.getOutMapNum();
	int x = layer.getMapSize().x;
	int y = layer.getMapSize().y;
	int nx = nextLayer.getMapSize().x;
	int ny = nextLayer.getMapSize().y;
	double** nextError;
	double** map;
	double** outMatrix, ** kroneckerMatrix;
	RectSize scale;
	outMatrix = new double* [x];
	kroneckerMatrix = new double* [x];
	for (int i = 0; i < x; i++)
	{
		outMatrix[i] = new double[y];
		kroneckerMatrix[i] = new double[y];
	}

	int lastmapNum = Lastlayer.getOutMapNum();
	int nextmapNum = nextLayer.getOutMapNum();

	double** thisError;

	for (int numBatch = 0; numBatch < batchSize; numBatch++)
	{
		for (int m = 0; m < mapNum; m++)
		{
			scale = layer.getScaleSize();
			thisError = layer.getError(numBatch, m);
			for (int ii = 0; ii < x; ii++) {
				for (int jj = 0; jj < y; jj++) {
					//printf("thisError[%d][%d][%d][%d]: %f\n", numBatch, m, ii, jj, thisError[ii][jj]);
				}
			}
		}
	}

	for (int numBatch = 0; numBatch < batchSize; numBatch++)
	{
		for (int m = 0; m < nextmapNum; m++)
		{

			nextError = nextLayer.getError(numBatch, m);
			for (int ii = 0; ii < nx; ii++) {
				for (int jj = 0; jj < ny; jj++) {
					//printf("nextError[%d][%d][%d][%d]: %f\n", numBatch, m, ii, jj, nextError[ii][jj]);
				}
			}
		}
	}
	//printf("================================================================================\n");

	int kernel_x = nextLayer.getKernelSize().x;
	int kernel_y = nextLayer.getKernelSize().y;
	double** nextkernel;

	for (int i = 0; i < mapNum; i++)
	{
		for (int j = 0; j < nextmapNum; j++)
		{
			nextkernel = nextLayer.getKernel(i, j);
			for (int ii = 0; ii < kernel_x; ii++)
			{
				for (int jj = 0; jj < kernel_y; jj++)
				{

					//printf("nextkernel[%d][%d][%d][%d]: %f\n", i, j, ii, jj, nextkernel[ii][jj]);
				}
			}
		}
	}
	//printf("================================================================================\n");


	double** derivativeOfActiveFun = new double* [batchSize];
	for (int i = 0; i < batchSize; i++)
	{
		derivativeOfActiveFun[i] = new double[mapNum];
	}

	for (int numBatch = 0; numBatch < batchSize; numBatch++)
	{
		for (int m = 0; m < mapNum; m++)
		{

			map = layer.outputmaps[numBatch][m];
			layer.setError(numBatch, m, outMatrix, x, y);


#if((SELECT_ACTIVE_FUNCTION == 1) || (SELECT_ACTIVE_FUNCTION == 2))
			//printf("derivative of sigmoid");
			//derivative of sigmoid
			matrixDsigmoidFChidden(map, x, y, &(derivativeOfActiveFun[numBatch][m]));//for sigmoid active fun. 20171201
#elif(SELECT_ACTIVE_FUNCTION == 3)
			//printf("derivative of ReLu");
			//derivative of ReLu
			matrixDreluFChidden(map, x, y, &(derivativeOfActiveFun[numBatch][m]));//for relu active fun.
#endif

		}
	}
	//printf("================================================================================\n");

	for (int numBatch = 0; numBatch < batchSize; numBatch++)
	{
		for (int m = 0; m < mapNum; m++)
		{
			thisError = layer.getError(numBatch, m);
			thisError[0][0] = derivativeOfActiveFun[numBatch][m];

			//printf("thisError[%d][%d][0][0]: %f\n", numBatch, m, thisError[0][0]);

		}
	}
	//printf("================================================================================\n");

	double** sumOflocalgradient = new double* [batchSize];
	for (int i = 0; i < batchSize; i++)
	{
		sumOflocalgradient[i] = new double[mapNum];
	}

	for (int i = 0; i < batchSize; i++)
	{
		for (int j = 0; j < mapNum; j++)
		{
			sumOflocalgradient[i][j] = 0.0;
		}
	}

	for (int numBatch = 0; numBatch < batchSize; numBatch++)
	{
		for (int i = 0; i < mapNum; i++)
		{
			for (int j = 0; j < nextmapNum; j++)
			{
				nextError = nextLayer.getError(numBatch, j);
				nextkernel = nextLayer.getKernel(i, j);

				sumOflocalgradient[numBatch][i] += nextError[0][0] * nextkernel[0][0];

			}

		}
	}

	for (int numBatch = 0; numBatch < batchSize; numBatch++)
	{
		for (int m = 0; m < mapNum; m++)
		{
			thisError = layer.getError(numBatch, m);
			if (0.0 == thisError[0][0])
			{
				thisError[0][0] = thisError[0][0] * sumOflocalgradient[numBatch][m];

				thisError[0][0] = abs(thisError[0][0]);
			}
			else {
				thisError[0][0] = thisError[0][0] * sumOflocalgradient[numBatch][m];
			}

			layer.setError(numBatch, m, thisError, 0, 0);

		}
	}


	for (int i = 0; i < x; i++)
	{
		delete[]outMatrix[i];
		delete[]kroneckerMatrix[i];
	}
	delete[]outMatrix;
	delete[]kroneckerMatrix;
}


void CNN::setHiddenLayerErrors()
{
	layers::iterator iter = m_layers.end();
	iter = iter - 2;
	for (iter; iter > m_layers.begin(); iter--)
	{
		switch ((*(iter)).getType())
		{
		case 'C':
			setConvErrors((*iter), (*(iter + 1)));
			break;
		case 'S':
			setSampErrors((*iter), (*(iter + 1)));
			break;
		case 'H':
			setFCHiddenLayerErrors((*(iter - 1)), (*iter), (*(iter + 1)));
			break;
		default:
			break;
		}
	}
}

void CNN::setSampErrors(Layer& layer, Layer& nextLayer)
{
	int mapNum = layer.getOutMapNum();
	int nextMapNum = nextLayer.getOutMapNum();
	int next_x = nextLayer.getMapSize().x;
	int next_y = nextLayer.getMapSize().y;
	int kernel_x = nextLayer.getKernelSize().x;
	int kernel_y = nextLayer.getKernelSize().y;
	int x = layer.getMapSize().x;
	int y = layer.getMapSize().y;
	double** nextError;
	double** kernel;
	double** sum, ** rotMatrix, ** sumNow;
	//initialize
	sum = new double* [x];
	for (int k = 0; k < x; k++)
	{
		sum[k] = new double[y];
	}
	rotMatrix = new double* [kernel_x];
	for (int kk = 0; kk < kernel_x; kk++)
	{
		rotMatrix[kk] = new double[kernel_y];
	}
	sumNow = new double* [x];
	for (int k = 0; k < x; k++)
	{
		sumNow[k] = new double[y];
	}
	double** extendMatrix;

	int m = next_x, n = next_y, km = kernel_x, kn = kernel_y;
	extendMatrix = new double* [m + 2 * (km - 1)];
	for (int k = 0; k < m + 2 * (km - 1); k++)
	{
		extendMatrix[k] = new double[n + 2 * (kn - 1)];
		for (int a = 0; a < n + 2 * (kn - 1); a++)
		{
			extendMatrix[k][a] = 0.0;
		}
	}
	//calculate
	for (int numBatch = 0; numBatch < batchSize; numBatch++)
	{
		for (int i = 0; i < mapNum; i++)
		{

			for (int j = 0; j < nextMapNum; j++)
			{

				nextError = nextLayer.getError(numBatch, j);
				kernel = nextLayer.getKernel(i, j);
				if (j == 0)
				{
					rot180(kernel, kernel_x, kernel_y, rotMatrix);
					convnFull(nextError, rotMatrix, next_x, next_y, kernel_x, kernel_y, sum, extendMatrix);

				}
				else
				{
					rot180(kernel, kernel_x, kernel_y, rotMatrix);
					convnFull(nextError, rotMatrix, next_x, next_y, kernel_x, kernel_y, sumNow, extendMatrix);
					ArrayPlus(sumNow, sum, x, y);

				}

			}
			layer.setError(numBatch, i, sum, x, y);
		}
	}
	for (int i = 0; i < x; i++)
	{
		delete[]sum[i];
		delete[]sumNow[i];
	}
	for (int i = 0; i < kernel_x; i++)
	{
		delete[]rotMatrix[i];
	}
	for (int i = 0; i < m + 2 * (km - 1); i++)
	{
		delete[]extendMatrix[i];
	}
	delete[]rotMatrix;
	delete[]sumNow;
	delete[]sum;
	delete[]extendMatrix;
}

void CNN::setConvErrors(Layer& layer, Layer& nextLayer)
{
	int mapNum = layer.getOutMapNum();
	int x = layer.getMapSize().x;
	int y = layer.getMapSize().y;
	int nx = nextLayer.getMapSize().x;
	int ny = nextLayer.getMapSize().y;
	double** nextError;
	double** map;
	double** outMatrix, ** kroneckerMatrix;
	RectSize scale;
	outMatrix = new double* [x];
	kroneckerMatrix = new double* [x];
	for (int i = 0; i < x; i++)
	{
		outMatrix[i] = new double[y];
		kroneckerMatrix[i] = new double[y];
	}
	for (int numBatch = 0; numBatch < batchSize; numBatch++)
	{
		for (int m = 0; m < mapNum; m++)
		{
			scale = nextLayer.getScaleSize();
			nextError = nextLayer.getError(numBatch, m);
			map = layer.outputmaps[numBatch][m];


#if((SELECT_ACTIVE_FUNCTION == 1) || (SELECT_ACTIVE_FUNCTION == 2))
			//printf("derivative of sigmoid");
			//derivative of sigmoid
			matrixDsigmoid(map, x, y, outMatrix);//for sigmoid active fun.
#elif(SELECT_ACTIVE_FUNCTION == 3)
			//printf("derivative of ReLu");
			//derivative of ReLu
			matrixDrelu(map, x, y, outMatrix);//for relu active fun.
#endif


			kronecker(nextError, scale, nx, ny, kroneckerMatrix);
			matrixMultiply(outMatrix, kroneckerMatrix, x, y);

			layer.setError(numBatch, m, outMatrix, x, y);

		}
	}
	for (int i = 0; i < x; i++)
	{
		delete[]outMatrix[i];
		delete[]kroneckerMatrix[i];
	}
	delete[]outMatrix;
	delete[]kroneckerMatrix;
}


void CNN::updateParas()
{
	layers::iterator iter = m_layers.begin();
	iter++;

	no_iter = 0;//begining at index 0 layer
	char str_file_kernel[1000];// initialized properly
	char str_file_bias[1000];// initialized properly

	for (iter; iter < m_layers.end(); iter++)
	{
		no_iter = no_iter + 1;
		sprintf(str_file_kernel, "./data/kernel_weight/kernel_weight_%d_%d", no_iter, (*iter).getType());
		sprintf(str_file_bias, "./data/bias/bias_%d_%d", no_iter, (*iter).getType());
		//printf("%s", str_file_kernel);

		switch ((*iter).getType())
		{
		case 'C':
			updateKernels(*iter, *(iter - 1), str_file_kernel, ETA_CONV, ALPHA_CONV);
			updateBias(*iter, str_file_bias, ETA_CONV);
			break;
		case 'H':
			updateKernels(*iter, *(iter - 1), str_file_kernel, ETA_FC, ALPHA_FC);
			updateBias(*iter, str_file_bias, ETA_FC);
			break;
		case 'O':
			updateKernels(*iter, *(iter - 1), str_file_kernel, ETA_FC, ALPHA_FC);
			updateBias(*iter, str_file_bias, ETA_FC);
			break;
		default:
			break;
		}
	}
}


void CNN::updateBias(Layer& layer, char* str_File_Bias, double eta)
{
	double**** errors = layer.errors;
	int mapNum = layer.getOutMapNum();
	int x = layer.getMapSize().x;
	int y = layer.getMapSize().y;
	double** error;
	error = new double* [x];
	for (int i = 0; i < x; i++)
	{
		error[i] = new double[y];
	}
	for (int j = 0; j < mapNum; j++)
	{
		sum(errors, j, x, y, batchSize, error);
		double deltaBias = sum(error, layer.getMapSize().x, layer.getMapSize().y) / batchSize;
		double bias = layer.bias[j] + eta * deltaBias;
		layer.bias[j] = bias;

		/***save bias***/
		if ((iterationsNum - 1) == iOfiterationsNum) {
			char str_file_bias_1[1000];
			sprintf(str_file_bias_1, "%s_%d.txt", str_File_Bias, j);
			FILE* fp_bias = fopen(str_file_bias_1, "w");

			fprintf(fp_bias, "%f ", layer.bias[j]);
			fprintf(fp_bias, "\n");

			fclose(fp_bias);
		}
	}
	for (int i = 0; i < x; i++)
	{
		delete[]error[i];
	}
	delete[]error;
}

void CNN::updateKernels(Layer& layer, Layer& lastLayer, char* str_File_Kernel, double eta, double alpha)
{
	int mapNum = layer.getOutMapNum();
	int lastMapNum = lastLayer.getOutMapNum();
	int kernel_x = layer.getKernelSize().x;
	int kernel_y = layer.getKernelSize().y;
	int last_x = lastLayer.getMapSize().x;
	int last_y = lastLayer.getMapSize().y;
	int x = layer.getMapSize().x;
	int y = layer.getMapSize().y;
	double** deltakernel1, ** deltakernel2, ** deltaNow;
	deltakernel1 = new double* [kernel_x];
	deltakernel2 = new double* [kernel_x];
	deltaNow = new double* [kernel_x];
	for (int ii = 0; ii < kernel_x; ii++)
	{
		deltakernel1[ii] = new double[kernel_y];
		deltakernel2[ii] = new double[kernel_y];
		deltaNow[ii] = new double[kernel_y];
	}
	for (int j = 0; j < mapNum; j++)
	{
		for (int i = 0; i < lastMapNum; i++)
		{
			for (int r = 0; r < batchSize; r++)
			{
				double** error = layer.errors[r][j];
				if (r == 0) {
					convnValid(lastLayer.outputmaps[r][i], error, last_x, last_y, x, y, deltakernel1);

				}
				else {
					convnValid(lastLayer.outputmaps[r][i], error, last_x, last_y, x, y, deltaNow);
					ArrayPlus(deltaNow, deltakernel1, kernel_x, kernel_y);

				}
			}
			setValue(deltakernel2, layer.laststepdeltakernel[i][j], layer.getKernelSize().x, layer.getKernelSize().y);//for adding momentum
			ArrayMultiply(deltakernel2, alpha, layer.getKernelSize().x, layer.getKernelSize().y);//for adding momentum
			ArrayPlus(deltakernel2, layer.kernel[i][j], layer.getKernelSize().x, layer.getKernelSize().y);//for adding momentum
			ArrayDivide(deltakernel1, batchSize, layer.getKernelSize().x, layer.getKernelSize().y);
			ArrayMultiply(deltakernel1, eta, layer.getKernelSize().x, layer.getKernelSize().y);
			setValue(layer.laststepdeltakernel[i][j], deltakernel1, layer.getKernelSize().x, layer.getKernelSize().y);//for adding momentum
			ArrayPlus(deltakernel1, layer.kernel[i][j], layer.getKernelSize().x, layer.getKernelSize().y);



			/***save kernel weight***/
			if ((iterationsNum - 1) == iOfiterationsNum) {
				char str_file_kernel_1[1000];
				sprintf(str_file_kernel_1, "%s_%d_%d.txt", str_File_Kernel, i, j);

				FILE* fp = fopen(str_file_kernel_1, "w");

				for (int mm = 0; mm < layer.getKernelSize().x; mm++)
				{
					for (int nn = 0; nn < layer.getKernelSize().y; nn++)
					{
						fprintf(fp, "%f ", layer.kernel[i][j][mm][nn]);
					}

				}
				fprintf(fp, "\n");
				fclose(fp);
			}

		}
	}
	for (int i = 0; i < kernel_x; i++)
	{
		delete[]deltakernel1[i];
		delete[]deltakernel2[i];
		delete[]deltaNow[i];
	}
	delete[]deltakernel1;
	delete[]deltakernel2;
	delete[]deltaNow;
}


void CNN::loadParas()
{
	layers::iterator iter = m_layers.begin();
	iter++;

	no_iter = 0;//begining at index 0 layer
	char str_file_kernel[1000];// initialized properly
	char str_file_bias[1000];// initialized properly

	for (iter; iter < m_layers.end(); iter++)
	{
		no_iter = no_iter + 1;
		sprintf(str_file_kernel, "./data/kernel_weight/kernel_weight_%d_%d", no_iter, (*iter).getType());
		sprintf(str_file_bias, "./data/bias/bias_%d_%d", no_iter, (*iter).getType());
		//printf("%s", str_file_kernel);

		switch ((*iter).getType())
		{
		case 'C':
			loadKernels(*iter, *(iter - 1), str_file_kernel);
			loadBias(*iter, str_file_bias);
			break;
		case 'H':
			loadKernels(*iter, *(iter - 1), str_file_kernel);
			loadBias(*iter, str_file_bias);
			break;
		case 'O':
			loadKernels(*iter, *(iter - 1), str_file_kernel);
			loadBias(*iter, str_file_bias);
			break;
		default:
			break;
		}
	}
}

void CNN::loadBias(Layer& layer, char* str_File_Bias)
{
	int mapNum = layer.getOutMapNum();
	int x = layer.getMapSize().x;
	int y = layer.getMapSize().y;

	for (int j = 0; j < mapNum; j++)
	{

		float bias;
		/***load bias***/
		char str_file_bias_1[1000];
		sprintf(str_file_bias_1, "%s_%d.txt", str_File_Bias, j);
		printf("%s\n", str_file_bias_1);
		FILE* fp_bias = fopen(str_file_bias_1, "r");

		fscanf(fp_bias, "%f ", &bias);
		layer.bias[j] = bias;

		printf("bias: %f\n", layer.bias[j]);
	}

}

void CNN::loadKernels(Layer& layer, Layer& lastLayer, char* str_File_Kernel)
{
	int mapNum = layer.getOutMapNum();
	int lastMapNum = lastLayer.getOutMapNum();
	int kernel_x = layer.getKernelSize().x;
	int kernel_y = layer.getKernelSize().y;
	int last_x = lastLayer.getMapSize().x;
	int last_y = lastLayer.getMapSize().y;
	int x = layer.getMapSize().x;
	int y = layer.getMapSize().y;

	for (int j = 0; j < mapNum; j++)
	{
		for (int i = 0; i < lastMapNum; i++)
		{
			/***load kernel weight***/
			char str_file_kernel_1[1000];
			sprintf(str_file_kernel_1, "%s_%d_%d.txt", str_File_Kernel, i, j);
			printf("%s\n", str_file_kernel_1);
			FILE* fp = fopen(str_file_kernel_1, "r");
			const size_t kernel_size_x = layer.getKernelSize().x;
			const size_t kernel_size_y = layer.getKernelSize().y;
			std::vector<float> vec_kernel;
			vec_kernel.reserve(kernel_size_x * kernel_size_y);

			for (int mm = 0; mm < kernel_size_x; mm++)
			{
				for (int nn = 0; nn < kernel_size_y; nn++)
				{
					//float kernel[kernel_size_x][kernel_size_y];

					fscanf(fp, "%f ", &(vec_kernel[((mm * kernel_size_x) + nn)]));
					layer.kernel[i][j][mm][nn] = vec_kernel[((mm * kernel_size_x) + nn)];
					printf("kernel: %f\n", layer.kernel[i][j][mm][nn]);
				}

			}

		}
	}
}


void CNN::test(double**** test_x, double** test_label, int number)
{
	cout << "Start test" << endl;
	//char _posfilepath[256]="Traing_p_";
	char _posfilepath[256] = "(Training)p_";
	char _negfilepath[256] = "Training_n_";
	char _imgfileextension[16] = ".bmp";
	char _input_posfilename[512];
	char _input_negfilename[512];
	int num_charaters_pos1;
	int num_charaters_neg1;

	int totalfause = 0, fause1 = 0, fause2 = 0, predict, real;
	int Num = number / batchSize;
	double**** Test;

	Test = new double*** [batchSize];

	FILE* fy_negfilename;
	fy_negfilename = fopen("./outputdata/error_predict_neg_filename.txt", "w");
	FILE* fy_posfilename;
	fy_posfilename = fopen("./outputdata/error_predict_pos_filename.txt", "w");
	for (int i = 0; i < Num; i++)
	{
		cout << "NO.of iteration(testing): " << i << endl;


		for (int j = 0; j < batchSize; j++)
		{
			cout << "NO.of batch(testing): " << j << endl;
			if (i == 0)
			{
				Test[j] = new double** [NumOfChannel];
			}
			for (int c = 0; c < NumOfChannel; c++)
			{

				if (i == 0)
				{
					Test[j][c] = new double* [IMG_H];
				}

				for (int m = 0; m < IMG_H; m++)
				{
					if (i == 0)
					{
						Test[j][c][m] = new double[IMG_W];
					}
					for (int n = 0; n < IMG_W; n++)
					{
						Test[j][c][m][n] = test_x[i * batchSize + j][c][m][n];
					}
				}
			}
		}

		forward(Test);
		layers::iterator iter = m_layers.end();
		iter--;
		for (int ii = 0; ii < batchSize; ii++)
		{
			cout << ii << endl;
			predict = findIndex((*iter).outputmaps[ii]);
			real = findIndex(test_label[i * batchSize + ii]);


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

	cout << "+++++++Finish test+++++++" << endl;
	cout << "Error predict number of neg: " << fause1 << endl;
	cout << "Error rate of neg: " << 1.0 * fause1 / number << endl;
	cout << "Error predict number of pos: " << fause2 << endl;
	cout << "Error rate of pos: " << 1.0 * fause2 / number << endl;
	cout << "Error predict total number: " << totalfause << endl;
	cout << "Total error rate: " << 1.0 * totalfause / number << endl << endl;

	FILE* fy;
	fy = fopen("./outputdata/fausePrun.txt", "a");
	/*
	if( (err=fopen_s(&fy, "fausePrun.txt", "a")) != 0 )
		exit(1) ;
	*/
	runi++;
	fprintf(fy, "epoch: %4d\n", runi);
	fprintf(fy, "neg: %4d %8f\n", fause1, 1.0 * fause1 / number);
	fprintf(fy, "pos: %4d %8f\n", fause2, 1.0 * fause2 / number);
	fprintf(fy, "total: %4d %8f\n\n", totalfause, 1.0 * totalfause / number);
	fclose(fy);
	fclose(fy_posfilename);
	fclose(fy_negfilename);

	for (int i = 0; i < batchSize; i++)
	{
		for (int c = 0; c < NumOfChannel; c++)
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
