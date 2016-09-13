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

using namespace std;
using namespace cv;
using namespace ml;

void ExtractData();
void setUpMtrices();
inline TermCriteria TC(int, double);
static void test_and_save_classifier(const Ptr<StatModel>&, int ntrain_samples, const string& filename_to_save);
static bool build_mlp_classifier(const string& filename_to_save);


Part p[33];
//Training Purposes
int labels[21] = { 0 };
float trainingData[21][64] = { 0 };

//Prediction Purposes
int pLabels[12] = { 0 };
float pData[12][64] = { 0 };

//Confusion Matrix
int conf;


int main(int argc, char * argv[]) {

	ExtractData();
	setUpMtrices();
	build_mlp_classifier("hello.txt");

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
				p[i].setResult(2);
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

	for (int i = 0; i < 21; i++)
	{
		for (int j = 0; j < 64; j++)
		{
			if (i < 7)
				trainingData[i][j] = tempTrainingData[i][j]; 
			else if (i >= 7 && i < 14)
				trainingData[i][j] = tempTrainingData[i + 3][j];
			else if (i >= 14)
				trainingData[i][j] = tempTrainingData[i + 6][j];
		}
		if (i < 7)
			labels[i] = tempLabels[i];
		else if (i >= 7 && i < 14)
			labels[i] = tempLabels[i + 3];
		else if (i >= 14)
			labels[i] = tempLabels[i + 6];
	}

	for (int i = 0; i < 12; i++)
	{
		for (int j = 0; j < 64; j++)
		{
			if (i < 3)
				pData[i][j] = tempTrainingData[i+7][j];
			else if (i >= 3 && i < 6)
				pData[i][j] = tempTrainingData[i+14][j];
			else if (i >= 6)
				pData[i][j] = tempTrainingData[i+21][j];
		}
		if (i < 3)
			pLabels[i] = tempLabels[i+7];
		else if (i >= 3 && i < 6)
			pLabels[i] = tempLabels[i+14];
		else if (i >= 6)
			pLabels[i] = tempLabels[i+21];
	}

	//for (int i = 27; i < 33; i++)
	//{
	//	for (int j = 0; j < 64; j++)
	//	{
	//		cout << tempTrainingData[i][j] << "-";
	//	}
	//	cout << tempLabels[i] << endl << endl;
	//}

	//cout << "------------------------------------------------" << endl  << endl;

	//for (int i = 6; i < 12; i++)
	//{
	//	for (int j = 0; j < 64; j++)
	//	{
	//		cout << pData[i][j] << "-";
	//	}
	//	cout << pLabels[i] << endl << endl;
	//}
}

static bool build_mlp_classifier(const string& filename_to_save)
{
	const int class_count = 26;	
	Mat train_data = Mat(21, 64, CV_32FC1, &trainingData);
	Mat responses = Mat(21, 1, CV_32S, &labels);

	Ptr<ANN_MLP> model;

	Mat train_responses = Mat::zeros(train_data.rows, class_count, CV_32F);

	// 1. unroll the responses
	cout << "Unrolling the responses...\n";
	for (int i = 0; i < train_data.rows; i++)
	{
		int cls_label = responses.at<int>(i);
		train_responses.at<float>(i, cls_label) = 1.f;
	}

	// 2. train classifier
	int layer_sz[] = { train_data.cols, 100, 100, class_count };
	int nlayers = (int)(sizeof(layer_sz) / sizeof(layer_sz[0]));
	Mat layer_sizes(1, nlayers, CV_32S, layer_sz);

#if 1
	int method = ANN_MLP::BACKPROP;
	double method_param = 0.001;
	int max_iter = 300;
#else
	int method = ANN_MLP::RPROP;
	double method_param = 0.1;
	int max_iter = 1000;
#endif

	Ptr<TrainData> tdata = TrainData::create(train_data, ROW_SAMPLE, train_responses);

	cout << "Training the classifier...\n";
	model = ANN_MLP::create();
	model->setLayerSizes(layer_sizes);
	model->setActivationFunction(ANN_MLP::SIGMOID_SYM, 0, 0);
	model->setTermCriteria(TC(max_iter, 0));
	model->setTrainMethod(method, method_param);
	model->train(tdata);
	cout << endl;

	test_and_save_classifier(model, train_data.rows, filename_to_save);

	return true;
}

inline TermCriteria TC(int iters, double eps)
{
	return TermCriteria(TermCriteria::MAX_ITER + (eps > 0 ? TermCriteria::EPS : 0), iters, eps);
}

static void test_and_save_classifier(const Ptr<StatModel>& model, int ntrain_samples, const string& filename_to_save)
{
	Mat pdata = Mat(12, 64, CV_32FC1, &pData);   //Loading the rest of the data for prediction
	int i, nsamples_all = pdata.rows;

	cout << "Actual Image Label\t|\tPredicted Label:" << endl;
	for (i = 0; i < nsamples_all; i++)
	{
		Mat sample = pdata.row(i);
		float r = model->predict(sample);
		cout << pLabels[i] << "\t\t\t|\t\t" << r << endl;
	}

	if (!filename_to_save.empty())
	{
		model->save(filename_to_save);
	}
}