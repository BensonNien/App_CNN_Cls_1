#include "CNNDataset.h"

void CNNDataset::Load(double**** train_x, double**** test_x, double** train_label, double** test_label)
{

	//read image by opencv

	char _posfilepath[256] = "./training_data/view007_false_missing/missing_object_64x32_rename/Training_p_";
	//char _posfilepath[256]="./training_data/nu_false_missing/missing_object_64x32_filter_rename/Training_p_";


	char _negfilepath[256] = "./training_data/view007_false_missing/false_object_64x32_filter_rename/Training_n_";
	//char _negfilepath[256]="./training_data/nu_false_missing/false_object_64x32_filter_rename2/Training_n_";

	char _imgfileextension[16] = ".bmp";
	char _input_posfilename[512];
	char _input_negfilename[512];

	Mat input_posimg1;
	Mat input_negimg1;

	int num_charaters_pos1;
	int num_charaters_neg1;

	int numOfPositiveCloning = 1;//clone positive sample


	for (int k = 0; k < 330; k++)//330
	{
		num_charaters_pos1 = sprintf(_input_posfilename, "%s%d%s", _posfilepath, k, _imgfileextension);
		printf("number of charaters: %d, string: \"%s\"\n", num_charaters_pos1, _input_posfilename);

		input_posimg1 = imread(_input_posfilename, IMREAD_COLOR);
		//imshow("input_posimg1", input_posimg1);
		//waitKey(0);

		for (int c = 0; c < 3; c++)
		{
			for (int i = 0; i < IMG_H; i++)
			{
				for (int j = 0; j < IMG_W; j++)
				{
					for (int indexOfPositiveCloning = 0; indexOfPositiveCloning < (2 * numOfPositiveCloning); indexOfPositiveCloning += 2)
					{

						train_x[((2 * numOfPositiveCloning) * k + 1) + indexOfPositiveCloning][c][i][j] = test_x[((2 * numOfPositiveCloning) * k + 1) + indexOfPositiveCloning][c][i][j] = (double)(input_posimg1.at<Vec3b>(i, j)[c]) / 255.0;
						train_label[((2 * numOfPositiveCloning) * k + 1) + indexOfPositiveCloning][0] = test_label[((2 * numOfPositiveCloning) * k + 1) + indexOfPositiveCloning][0] = 0.0;
						train_label[((2 * numOfPositiveCloning) * k + 1) + indexOfPositiveCloning][1] = test_label[((2 * numOfPositiveCloning) * k + 1) + indexOfPositiveCloning][1] = 1.0;


					}
				}
			}
		}

	}


	int numOfNegativeCloning = 22;//clone negative sample //22
	for (int k = 0; k < 15; k++)//15
	{
		num_charaters_neg1 = sprintf(_input_negfilename, "%s%d%s", _negfilepath, k, _imgfileextension);
		printf("number of charaters: %d, string: \"%s\"\n", num_charaters_neg1, _input_negfilename);

		input_negimg1 = imread(_input_negfilename, IMREAD_COLOR);
		//imshow("input_negimg1", input_negimg1);
		//waitKey(0);

		for (int c = 0; c < 3; c++)
		{
			for (int i = 0; i < IMG_H; i++)
			{
				for (int j = 0; j < IMG_W; j++)
				{

					for (int indexOfNegativeCloning = 0; indexOfNegativeCloning < (2 * numOfNegativeCloning); indexOfNegativeCloning += 2)
					{

						train_x[((2 * numOfNegativeCloning) * k) + indexOfNegativeCloning][c][i][j] = test_x[((2 * numOfNegativeCloning) * k) + indexOfNegativeCloning][c][i][j] = (double)(input_negimg1.at<Vec3b>(i, j)[c]) / 255.0;
						train_label[((2 * numOfNegativeCloning) * k) + indexOfNegativeCloning][0] = test_label[((2 * numOfNegativeCloning) * k) + indexOfNegativeCloning][0] = 1.0;
						train_label[((2 * numOfNegativeCloning) * k) + indexOfNegativeCloning][1] = test_label[((2 * numOfNegativeCloning) * k) + indexOfNegativeCloning][1] = 0.0;


					}

				}
			}
		}
	}


}
