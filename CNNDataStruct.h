#pragma once

#include <iostream>
#include <string>

class RectSize
{
public:
	int x;
	int y;

	RectSize()
	{
		this->x = 0;
		this->y = 0;
	}
	~RectSize() {};

	RectSize(int x, int y)
	{
		this->x = x;
		this->y = y;
	}

	RectSize divide(RectSize scalesize)
	{
		int x = this->x / scalesize.x;
		int y = this->y / scalesize.y;
		if (x * scalesize.x != this->x || y * scalesize.y != this->y)
		{
			std::cout << this << "can not divide" << std::endl;
		}
		return RectSize(x, y);
	}

	RectSize substract(RectSize s, int append)
	{
		int x = this->x - s.x + append;
		int y = this->y - s.y + append;
		return RectSize(x, y);
	}

};

struct DatasetLoadingParamPKG {
	float**** p_train_images_ = nullptr;
	float**** p_test_images_ = nullptr;
	float** p_train_labels_ = nullptr;
	float** p_test_labels_ = nullptr;
	int num_pos_images_ = 0;
	int num_neg_images_ = 0;
	int channels_image_ = 0;
	std::string pos_images_root_path_;
	std::string neg_images_root_path_;
	std::string images_ext_;

	DatasetLoadingParamPKG(float**** p_train_images, float**** p_test_images, 
		float** p_train_labels, float** p_test_labels, int num_pos_images, int num_neg_images, int channels_image,
		std::string pos_images_root_path, std::string neg_images_root_path, std::string images_ext) 
	{

		p_train_images_ = p_train_images;
		p_test_images_ = p_test_images;
		p_train_labels_ = p_train_labels;
		p_test_labels_ = p_test_labels;
		num_pos_images_ = num_pos_images;
		num_neg_images_ = num_neg_images;
		channels_image_ = channels_image;

		pos_images_root_path_ = pos_images_root_path;
		neg_images_root_path_ = neg_images_root_path;
		images_ext_ = images_ext;

	}

};
