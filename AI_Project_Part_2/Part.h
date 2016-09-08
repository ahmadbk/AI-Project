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

class Part {

private:
	string name;

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

};