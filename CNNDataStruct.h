#pragma once

#include <iostream>

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
