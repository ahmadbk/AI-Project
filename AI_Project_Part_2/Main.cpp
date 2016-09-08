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


using namespace std;
using namespace cv;
Part p[33];
float labels[33] = { 0 };
float trainingData[33][48] = { 0 };


int main(int argc, char * argv[]) {

	ExtractData();

	for (int i = 0; i < 33; i++)
		labels[i] = p[i].getResult();

	for (int i = 0; i < 33; i++)
	{
		int k = 0;
		for (int j = 0; j < 8; j++)
		{
			trainingData[i][k++] = p[i].partData[j].getDistance();
			trainingData[i][k++] = p[i].partData[j].getOrientation();
			trainingData[i][k++] = p[i].partData[j].f.getContrast();
			trainingData[i][k++] = p[i].partData[j].f.getCorrelation();
			trainingData[i][k++] = p[i].partData[j].f.getEnergy();
			trainingData[i][k++] = p[i].partData[j].f.getEntropy();
			trainingData[i][k++] = p[i].partData[j].f.getHomogeneity();
			trainingData[i][k++] = p[i].partData[j].f.getMaxProb();
		}
	}

	for (int i = 0; i < 33; i++)
	{
		for (int j = 0; j < 48; j++)
		{
			cout << trainingData[i][j] << "|";
		}
		cout << endl;
	}

	Ptr<ml::SVM> svm = ml::SVM::create();

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
				p[i].setType(-1);
			else if (imageName.find("good") != string::npos)
				p[i].setType(1);
			else if(imageName.find("empty") != string::npos)
				p[i].setType(0);

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

