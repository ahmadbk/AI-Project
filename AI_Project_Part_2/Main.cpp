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
#include "Part.h"


void ExtractData();
void setUpMtrices();


using namespace std;
using namespace cv;
Part p[33];
int labels[12] = { 0 };
float trainingData[12][64] = { 0 };
int pLabels[21] = { 0 };
float pData[21][64] = { 0 };


int main(int argc, char * argv[]) {

	ExtractData();
	setUpMtrices();
	system("pause");

}

void ExtractData()
{
	std::ifstream file;
	string line;
	file.open("C:/Users/ahmadbk/Desktop/Semester1/Artificial Intelligence/Practical 1/213504260/Source Code/AI-Project/Results.txt");
	if (!file.is_open())
	{
		cout << "Failed To Open File";
	}
	else
	{
		int i = 0;
		int k = 0;

		while (getline(file, line))
		{
			char * line1 = &line[0];
			char *next_token1 = NULL;
			char *next_token2 = NULL;
			char *token1 = NULL;
			char *token2 = NULL;
			token1 = strtok_s(line1, ";", &next_token1);
			string imageName = token1;

			p[i].setName(imageName);
			
			if (imageName.find("bad") != string::npos)
				p[i].setResult(-1);
			else if (imageName.find("good") != string::npos)
				p[i].setResult(1);
			else if(imageName.find("empty") != string::npos)
				p[i].setResult(0);

			token1 = strtok_s(NULL, ";", &next_token1);

			while (token1 != NULL)
			{
				if (token1 != NULL)
				{
					token2 = strtok_s(token1, "#", &next_token2);//distance
					if (token2 != NULL)
						p[i].partData[k].setDistance(atof(token2));

					token2 = strtok_s(NULL, "#", &next_token2);//orientation
					if (token2 != NULL)
						p[i].partData[k].setOrientation(atof(token2));

					token2 = strtok_s(NULL, "#", &next_token2);//Max Probability
					if (token2 != NULL)
						p[i].partData[k].f.setMaxProb(atof(token2));

					token2 = strtok_s(NULL, "#", &next_token2);//Energy
					if (token2 != NULL)
						p[i].partData[k].f.setEnergy(atof(token2));

					token2 = strtok_s(NULL, "#", &next_token2);//Homogeneity
					if (token2 != NULL)
						p[i].partData[k].f.setHomogeneity(atof(token2));

					token2 = strtok_s(NULL, "#", &next_token2);//Contrast
					if (token2 != NULL)
						p[i].partData[k].f.setContrast(atof(token2));

					token2 = strtok_s(NULL, "#", &next_token2);//Correlation
					if (token2 != NULL)
						p[i].partData[k].f.setCorrelation(atof(token2));

					token2 = strtok_s(NULL, "#", &next_token2);//Entropy
					if (token2 != NULL)
						p[i].partData[k].f.setEntropy(atof(token2));

				}
				token1 = strtok_s(NULL, ";", &next_token1);
				k++;
			}
			k = 0;
			i++;
		}

		//for (int j = 0; j < 33; j++)
		//{
		//	cout << p[j].getName() << ":" << p[j].getResult() << endl;
		//	for (int k = 0; k < 8; k++)
		//	{
		//		cout << p[j].partData[k].getDistance() << ":" << p[j].partData[k].getOrientation() << ":";
		//		cout << p[j].partData[k].f.getMaxProb() << ":" << p[j].partData[k].f.getEnergy() << ":" << p[j].partData[k].f.getHomogeneity() << ":" << p[j].partData[k].f.getContrast() << ":" << p[j].partData[k].f.getCorrelation() << ":" << p[j].partData[k].f.getEntropy() << endl;
		//	}
		//	cout << endl;
		//}
	}

	file.close();

}

void setUpMtrices()
{
	int tempLabels[33] = { 0 };
	float tempTrainingData[33][64] = { 0 };

	for (int i = 0; i < 33; i++)
		tempLabels[i] = p[i].getResult();
	

	for (int i = 0; i < 33; i++)
	{
		int k = 0;
		for (int j = 0; j < 8; j++)
		{
			tempTrainingData[i][k++] = p[i].partData[j].getDistance();
			tempTrainingData[i][k++] = p[i].partData[j].getOrientation();
			tempTrainingData[i][k++] = p[i].partData[j].f.getContrast();
			tempTrainingData[i][k++] = p[i].partData[j].f.getCorrelation();
			tempTrainingData[i][k++] = p[i].partData[j].f.getEnergy();
			tempTrainingData[i][k++] = p[i].partData[j].f.getEntropy();
			tempTrainingData[i][k++] = p[i].partData[j].f.getHomogeneity();
			tempTrainingData[i][k++] = p[i].partData[j].f.getMaxProb();
		}
	}

	for (int i = 0; i < 12; i++)
	{
		for (int j = 0; j < 64; j++)
		{
			if (i < 4)
				trainingData[i][j] = tempTrainingData[i][j]; 
			else if (i >= 4 && i < 8)
				trainingData[i][j] = tempTrainingData[i + 6][j];
			else if (i >= 8)
				trainingData[i][j] = tempTrainingData[i + 12][j];
		}
		if (i < 4)
			labels[i] = tempLabels[i]; 
		else if (i >= 4 && i < 8)
			labels[i] = tempLabels[i + 6];
		else if (i >= 8)
			labels[i] = tempLabels[i + 14];		
	}

	for (int i = 0; i < 21; i++)
	{
		for (int j = 0; j < 64; j++)
		{
			if (i < 6)
				pData[i][j] = tempTrainingData[i+4][j];
			else if (i >= 6 && i < 12)
				pData[i][j] = tempTrainingData[i+8][j];
			else if (i >= 12)
				pData[i][j] = tempTrainingData[i+12][j];
		}
		if (i < 6)
			pLabels[i] = tempLabels[i+4]; 
		else if (i >= 6 && i < 12)
			pLabels[i] = tempLabels[i+8];
		else if (i >= 12)
			pLabels[i] = tempLabels[i+12];
	}
}


//Ptr<ml::SVM> svm = ml::SVM::create();

//// Data for visual representation
//int width = 512, height = 512;
//Mat image = Mat::zeros(height, width, CV_8UC3);

//// Set up training data
//Mat labelsMat(33, 1, CV_32S, labels);
//Mat trainingDataMat(33, 64, CV_32FC1, trainingData);

//// Train the SVM
//svm->setType(ml::SVM::C_SVC);
//svm->setKernel(ml::SVM::LINEAR);
//svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-7));
////svm->setGamma(3);
//svm->train(trainingDataMat, ml::ROW_SAMPLE, labelsMat);

//// Show the decision regions given by the SVM
//Vec3b green(0, 255, 0), blue(255, 0, 0) , red(0, 0, 255);

///*
//Mat sampleMat = (Mat_<float>(1, 64) << 1,0,0.0556522,0.0210734,0.629576,3.97587,0.861576,4.33083,
//									   1,45,0.0498361,0.010869,0.429037,11.1753,0.612005,4.87243,
//									   1,90,0.091425,0.0176644,0.531528,8.10971,0.735974,4.62668,
//									   1,135,0.0500546,0.0108245,0.427532,11.2877,0.608104,4.87383,
//									   3,0,0.040879,0.0118639,0.462,13.5043,0.480707,4.80591,
//									   3,45,0.0199375,0.00734851,0.304332,24.1898,0.0622217,5.0946,
//								       3,90,0.0896728,0.0152826,0.443339,14.2769,0.532541,4.8087,
//									   3,135,0.0184963,0.00739352,0.307927,23.7264,0.0805686,5.09004);//good image9
//*/

//for (int i = 0; i < image.rows; ++i)
//	for (int j = 0; j < image.cols; ++j)
//	{
//		Mat sampleMat = (Mat_<float>(1, 64) << 1, 0, 0.0556522, 0.0210734, 0.629576, 3.97587, 0.861576, 4.33083, 1, 45, 0.0498361, 0.010869, 0.429037, 11.1753, 0.612005, 4.87243, 1, 90, 0.091425, 0.0176644, 0.531528, 8.10971, 0.735974, 4.62668, 1, 135, 0.0500546, 0.0108245, 0.427532, 11.2877, 0.608104, 4.87383, 3, 0, 0.040879, 0.0118639, 0.462, 13.5043, 0.480707, 4.80591, 3, 45, 0.0199375, 0.00734851, 0.304332, 24.1898, 0.0622217, 5.0946, 3, 90, 0.0896728, 0.0152826, 0.443339, 14.2769, 0.532541, 4.8087, 3, 135, 0.0184963, 0.00739352, 0.307927, 23.7264, 0.0805686, 5.09004);//good image
//		//Mat sampleMat = (Mat_<float>(1, 64) << 1,0,0.0568401,0.019672,0.597911,3.40824,0.825642,4.28173,1,45,0.0399709,0.0134097,0.501062,6.15044,0.681509,4.58679,1,90,0.0465116,0.021869,0.6548,3.05325,0.841502,4.17655,1,135,0.0414244,0.0143955,0.518552,5.50169,0.715103,4.52574,3,0,0.038886,0.0119066,0.45043,10.6285,0.426248,4.72859,3,45,0.0280749,0.00992875,0.373532,15.5618,0.115914,4.86259,3,90,0.0298824,0.0132436,0.503332,8.02729,0.570005,4.622,3,135,0.0219251,0.0100109,0.386123,13.0439,0.259723,4.83616);//bad image
//		float response = svm->predict(sampleMat);
//		if (response == 1.0)
//			image.at<Vec3b>(i, j) = green;
//		else if (response == -1.0)
//			image.at<Vec3b>(i, j) = blue;
//		else if (response == 0.0)
//			image.at<Vec3b>(i, j) = red;
//	}

//imwrite("result.png", image);        // save the image
//imshow("SVM Result", image); // show it to the user
//waitKey(0);