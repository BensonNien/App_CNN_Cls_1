#include "CNNDataset.h"

//void CNNDataset::Load(float**** train_x, float**** test_x, float** train_label, float** test_label)
//{
//
//	//read image by opencv
//
//	char _posfilepath[256]=".\\Pedestrian_TrainingDataset_PNG\\64x32_part_balance_v2\\pos\\Training_p_";
//	char _negfilepath[256]=".\\Pedestrian_TrainingDataset_PNG\\64x32_part_balance_v2\\neg\\Training_n_";
//
//	char _imgfileextension[16] = ".png";
//	char _input_posfilename[512];
//	char _input_negfilename[512];
//
//	Mat input_posimg1;
//	Mat input_negimg1;
//
//	int num_charaters_pos1;
//	int num_charaters_neg1;
//
//	int num_pos_clone = 1;//clone positive sample
//
//
//	for (int k = 0; k < 1000; k++)//330
//	{
//		num_charaters_pos1 = sprintf(_input_posfilename, "%s%d%s", _posfilepath, k, _imgfileextension);
//		printf("number of charaters: %d, string: \"%s\"\n", num_charaters_pos1, _input_posfilename);
//
//		input_posimg1 = imread(_input_posfilename, IMREAD_COLOR);
//		//imshow("input_posimg1", input_posimg1);
//		//waitKey(0);
//
//		for (int c = 0; c < 3; c++)
//		{
//			for (int i = 0; i < IMG_H; i++)
//			{
//				for (int j = 0; j < IMG_W; j++)
//				{
//					for (int idx_pos_clone = 0; idx_pos_clone < (2 * num_pos_clone); idx_pos_clone += 2)
//					{
//
//						train_x[((2 * num_pos_clone) * k + 1) + idx_pos_clone][c][i][j] = test_x[((2 * num_pos_clone) * k + 1) + idx_pos_clone][c][i][j] = (float)(input_posimg1.at<Vec3b>(i, j)[c]) / 255.0;
//						train_label[((2 * num_pos_clone) * k + 1) + idx_pos_clone][0] = test_label[((2 * num_pos_clone) * k + 1) + idx_pos_clone][0] = 0.0;
//						train_label[((2 * num_pos_clone) * k + 1) + idx_pos_clone][1] = test_label[((2 * num_pos_clone) * k + 1) + idx_pos_clone][1] = 1.0;
//
//
//					}
//				}
//			}
//		}
//
//	}
//
//
//	int num_neg_clone = 1;//clone negative sample //22
//	for (int k = 0; k < 1000; k++)//15
//	{
//		num_charaters_neg1 = sprintf(_input_negfilename, "%s%d%s", _negfilepath, k, _imgfileextension);
//		printf("number of charaters: %d, string: \"%s\"\n", num_charaters_neg1, _input_negfilename);
//
//		input_negimg1 = imread(_input_negfilename, IMREAD_COLOR);
//		//imshow("input_negimg1", input_negimg1);
//		//waitKey(0);
//
//		for (int c = 0; c < 3; c++)
//		{
//			for (int i = 0; i < IMG_H; i++)
//			{
//				for (int j = 0; j < IMG_W; j++)
//				{
//
//					for (int indexOfNegativeCloning = 0; indexOfNegativeCloning < (2 * num_neg_clone); indexOfNegativeCloning += 2)
//					{
//
//						train_x[((2 * num_neg_clone) * k) + indexOfNegativeCloning][c][i][j] = test_x[((2 * num_neg_clone) * k) + indexOfNegativeCloning][c][i][j] = (float)(input_negimg1.at<Vec3b>(i, j)[c]) / 255.0;
//						train_label[((2 * num_neg_clone) * k) + indexOfNegativeCloning][0] = test_label[((2 * num_neg_clone) * k) + indexOfNegativeCloning][0] = 1.0;
//						train_label[((2 * num_neg_clone) * k) + indexOfNegativeCloning][1] = test_label[((2 * num_neg_clone) * k) + indexOfNegativeCloning][1] = 0.0;
//
//
//					}
//
//				}
//			}
//		}
//	}
//
//
//}

void CNNDataset::Load(DatasetLoadingParamPKG& r_dataset_param)
{
	//read image by opencv

	int num_pos_clone = 1;//clone positive sample
	for (int k = 0; k < r_dataset_param.num_pos_images_; k++)//330
	{
		std::string input_pos_image_path = r_dataset_param.pos_images_root_path_ + to_string(k) + r_dataset_param.images_ext_;
		std::cout << "input_pos_image_path: " + input_pos_image_path << std::endl;

		Mat input_posimg1 = imread(input_pos_image_path, IMREAD_COLOR);
		//imshow("input_posimg1", input_posimg1);
		//waitKey(0);

		for (int c = 0; c < r_dataset_param.channels_image_; c++)
		{
			for (int i = 0; i < IMG_H; i++)
			{
				for (int j = 0; j < IMG_W; j++)
				{
					for (int idx_pos_clone = 0; idx_pos_clone < (2 * num_pos_clone); idx_pos_clone += 2)
					{
						r_dataset_param.p_train_images_[((2 * num_pos_clone) * k + 1) + idx_pos_clone][c][i][j] = r_dataset_param.p_test_images_[((2 * num_pos_clone) * k + 1) + idx_pos_clone][c][i][j] = (float)(input_posimg1.at<Vec3b>(i, j)[c]) / 255.0;
						r_dataset_param.p_train_labels_[((2 * num_pos_clone) * k + 1) + idx_pos_clone][0] = r_dataset_param.p_test_labels_[((2 * num_pos_clone) * k + 1) + idx_pos_clone][0] = 0.0;
						r_dataset_param.p_train_labels_[((2 * num_pos_clone) * k + 1) + idx_pos_clone][1] = r_dataset_param.p_test_labels_[((2 * num_pos_clone) * k + 1) + idx_pos_clone][1] = 1.0;
					}
				}
			}
		}

	}


	int num_neg_clone = 1;//clone negative sample //22
	for (int k = 0; k < r_dataset_param.num_neg_images_; k++)//15
	{
		std::string input_neg_image_path = r_dataset_param.neg_images_root_path_ + to_string(k) + r_dataset_param.images_ext_;
		std::cout << "input_neg_image_path: " + input_neg_image_path << std::endl;

		Mat input_negimg1 = imread(input_neg_image_path, IMREAD_COLOR);
		//imshow("input_negimg1", input_negimg1);
		//waitKey(0);

		for (int c = 0; c < r_dataset_param.channels_image_; c++)
		{
			for (int i = 0; i < IMG_H; i++)
			{
				for (int j = 0; j < IMG_W; j++)
				{

					for (int idx_neg_clone = 0; idx_neg_clone < (2 * num_neg_clone); idx_neg_clone += 2)
					{
						r_dataset_param.p_train_images_[((2 * num_neg_clone) * k) + idx_neg_clone][c][i][j] = r_dataset_param.p_test_images_[((2 * num_neg_clone) * k) + idx_neg_clone][c][i][j] = (float)(input_negimg1.at<Vec3b>(i, j)[c]) / 255.0;
						r_dataset_param.p_train_labels_[((2 * num_neg_clone) * k) + idx_neg_clone][0] = r_dataset_param.p_test_labels_[((2 * num_neg_clone) * k) + idx_neg_clone][0] = 1.0;
						r_dataset_param.p_train_labels_[((2 * num_neg_clone) * k) + idx_neg_clone][1] = r_dataset_param.p_test_labels_[((2 * num_neg_clone) * k) + idx_neg_clone][1] = 0.0;
					}

				}
			}
		}
	}


}
