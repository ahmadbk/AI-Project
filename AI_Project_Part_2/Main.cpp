//Ahmad Bin Khalid
//Imran Paruk

#include<Windows.h>
#include <vector>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<fstream>
#include<iostream>
#include<String.h>
#include<ctime>
#include "opencv2/ml.hpp"
#include "Features.h"


using namespace std;
using namespace cv;

int main(int argc, char * argv[]) {

	std::ifstream file;
	string line;
	file.open("C:/Users/ahmadbk/Desktop/Semester1/Artificial Intelligence/Practical 1/213504260/Source Code/AI-Project/Results.txt");
	if (!file.is_open())
	{
		cout << "Failed To Open File";
		return 1;
	}
	else
	{
		while (getline(file, line, ';'))
		{
			cout << line << endl;
		}
	}

	file.close();


	system("pause");


	Ptr<ml::SVM> svm = ml::SVM::create();

}