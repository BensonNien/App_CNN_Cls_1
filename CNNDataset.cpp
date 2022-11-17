#include "CNNDataset.h"

void CNNDataset::Load(DatasetLoadingParamPKG& r_dataset_param)
{
	//read image by opencv

	size_t num_pos_clone = 1;//clone positive sample
	for (size_t k = 0; k < r_dataset_param.num_pos_images_; k++)//330
	{
		std::string input_pos_image_path = r_dataset_param.pos_images_root_path_ + to_string(k) + r_dataset_param.images_ext_;
		std::cout << "input_pos_image_path: " + input_pos_image_path << std::endl;

		Mat input_posimg1 = imread(input_pos_image_path, IMREAD_COLOR);
		//imshow("input_posimg1", input_posimg1);
		//waitKey(0);

		size_t channels_image = r_dataset_param.channels_image_;
		size_t rows_image = r_dataset_param.rows_image_;
		size_t cols_image = r_dataset_param.cols_image_;

		size_t shift_idx_clone_image = 0;
		size_t shift_idx_channel_clone_image = 0;
		size_t idx_clone_image = 0;

		size_t shift_idx_clone_label = 0;

		for (size_t c = 0; c < channels_image; c++)
		{
			for (size_t i = 0; i < rows_image; i++)
			{
				for (size_t j = 0; j < cols_image; j++)
				{
					for (size_t idx_pos_clone = 0; idx_pos_clone < (2 * num_pos_clone); idx_pos_clone += 2)
					{
						shift_idx_clone_image = (((2 * num_pos_clone) * k + 1) + idx_pos_clone) * channels_image * rows_image * cols_image;
						shift_idx_channel_clone_image = c * rows_image * cols_image;
						idx_clone_image = shift_idx_clone_image + shift_idx_channel_clone_image + ((i * cols_image) + j);
						shift_idx_clone_label = (((2 * num_pos_clone) * k + 1) + idx_pos_clone) * r_dataset_param.num_output_cls_;
						r_dataset_param.vec_images_.at(idx_clone_image) = (float)(input_posimg1.at<Vec3b>(i, j)[c]) / 255.0;
						r_dataset_param.vec_labels_.at(shift_idx_clone_label + 0) = 0.0;
						r_dataset_param.vec_labels_.at(shift_idx_clone_label + 1) = 1.0;
					}
				}
			}
		}

	}


	size_t num_neg_clone = 1;//clone negative sample //22
	for (size_t k = 0; k < r_dataset_param.num_neg_images_; k++)//15
	{
		std::string input_neg_image_path = r_dataset_param.neg_images_root_path_ + to_string(k) + r_dataset_param.images_ext_;
		std::cout << "input_neg_image_path: " + input_neg_image_path << std::endl;

		Mat input_negimg1 = imread(input_neg_image_path, IMREAD_COLOR);
		//imshow("input_negimg1", input_negimg1);
		//waitKey(0);

		size_t channels_image = r_dataset_param.channels_image_;
		size_t rows_image = r_dataset_param.rows_image_;
		size_t cols_image = r_dataset_param.cols_image_;

		size_t shift_idx_clone_image = 0;
		size_t shift_idx_channel_clone_image = 0;
		size_t idx_clone_image = 0;

		size_t shift_idx_clone_label = 0;

		for (size_t c = 0; c < channels_image; c++)
		{
			for (size_t i = 0; i < rows_image; i++)
			{
				for (size_t j = 0; j < cols_image; j++)
				{
					for (size_t idx_neg_clone = 0; idx_neg_clone < (2 * num_neg_clone); idx_neg_clone += 2)
					{

						shift_idx_clone_image = (((2 * num_neg_clone) * k) + idx_neg_clone) * channels_image * rows_image * cols_image;
						shift_idx_channel_clone_image = c * rows_image * cols_image;
						idx_clone_image = shift_idx_clone_image + shift_idx_channel_clone_image + ((i * cols_image) + j);
						shift_idx_clone_label = (((2 * num_neg_clone) * k) + idx_neg_clone) * r_dataset_param.num_output_cls_;
						r_dataset_param.vec_images_.at(idx_clone_image) = (float)(input_negimg1.at<Vec3b>(i, j)[c]) / 255.0;
						r_dataset_param.vec_labels_.at(shift_idx_clone_label + 0) = 1.0;
						r_dataset_param.vec_labels_.at(shift_idx_clone_label + 1) = 0.0;

					}

				}
			}
		}
	}


}
