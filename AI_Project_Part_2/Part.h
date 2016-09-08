#pragma once

#include<Windows.h>
#include <vector>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<fstream>
#include<iostream>
#include<String.h>
#include<ctime>
#include "Features.h"
#include "Set.h"

using namespace std;

//type: 1:good 0:empty -1:bad

class Part {

private:
	string name;
	int result;

public:

	Set partData[8];

	string getName()
	{
		return name;
	}

	void setName(string n)
	{
		name = n;
	}

	int getResult()
	{
		return result;
	}

	void setResult(int t)
	{
		result = t;
	}

};