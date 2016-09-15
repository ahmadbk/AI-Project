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
static void predict_display_results(const Ptr<StatModel>&, const string& filename_to_save);
static bool train_mlp_classifier(const string& filename_to_save);


Part p[33];
//Training Purposes
int labels[21] = { 0 };
float trainingData[21][48] = { 0 };

//Prediction Purposes
int pLabels[12] = { 0 };
float pData[12][48] = { 0 };

//Confusion Matrix
int confGood[2][2];
int confBad[2][2];
int confEmpty[2][2];
//int confFinal[2][2];


int main(int argc, char * argv[]) {

	ExtractData();	//Extract Data from textfile
	setUpMtrices();	//Split data into appropriate train and test matrices
	train_mlp_classifier("classifier.txt");	//Initialize and train neural network
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
	float tempTrainingData[33][48] = { 0 };

	for (int i = 0; i < 33; i++)
		tempLabels[i] = p[i].getResult();
	

	for (int i = 0; i < 33; i++)
	{
		int k = 0;
		for (int j = 0; j < 8; j++)
		{
			//tempTrainingData[i][k++] = p[i].partData[j].getDistance();
			//tempTrainingData[i][k++] = p[i].partData[j].getOrientation();
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
		for (int j = 0; j < 48; j++)
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
		for (int j = 0; j < 48; j++)
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

static bool train_mlp_classifier(const string& filename_to_save)
{
	const int class_count = 3;	
	Mat train_data = Mat(21, 48, CV_32FC1, &trainingData);
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
	int layer_sz[] = { train_data.cols,48,class_count };
	int nlayers = (int)(sizeof(layer_sz) / sizeof(layer_sz[0]));
	Mat layer_sizes(1, nlayers, CV_32S, layer_sz);

#if 1
	int method = ANN_MLP::BACKPROP;
	double method_param = 0.000001;
	int max_iter = 100;
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

	predict_display_results(model, filename_to_save);

	return true;
}

inline TermCriteria TC(int iters, double eps)
{
	return TermCriteria(TermCriteria::MAX_ITER + (eps > 0 ? TermCriteria::EPS : 0), iters, eps);
}

static void predict_display_results(const Ptr<StatModel>& model, const string& filename_to_save)
{
	Mat pdata = Mat(12, 48, CV_32FC1, &pData);   //Loading the rest of the data for prediction
	int i, nsamples_all = pdata.rows;

	float fp_rate_good, tp_rate_good;
	float accuracy_good, precision_good, f_score_good;
	float fp_rate_bad, tp_rate_bad;
	float accuracy_bad, precision_bad, f_score_bad;
	float fp_rate_empty, tp_rate_empty;
	float accuracy_empty, precision_empty, f_score_empty;
	float accuracy_final, precision_final, f_score_final;

	int totalClassGood = 0;
	int classGoodCorrect = 0;
	int totalClassBad = 0;
	int classBadCorrect = 0;
	int totalClassEmpty = 0;
	int classEmptyCorrect = 0;

	for (i = 0; i < nsamples_all; i++)
	{
		Mat sample = pdata.row(i);
		float r = model->predict(sample);
		if (pLabels[i] == 1)
		{
			totalClassGood++;
			if (r == 1)
				classGoodCorrect++;
		}
		else if (pLabels[i] == 0)
		{
			totalClassEmpty++;
			if (r == 0)
				classEmptyCorrect++;
		}
		else
		{
			totalClassBad++;
			if (r == 2)
				classBadCorrect++;
		}

//----------------------------------------------------------------------------------

//Good vs Non-Good
//----------------------------------------------------------------------------------
		if (r == 1)
		{
			if (pLabels[i] == 1)
				confGood[0][0]++;//True Positives
			else
				confGood[0][1]++;//False posiive
		}
		else
		{
			if (pLabels[i] == 1)
				confGood[1][0]++;//False negative
			else
				confGood[1][1]++;//True Negatives
		}
//----------------------------------------------------------------------------------

//Empty vs NonEmpty
//----------------------------------------------------------------------------------
		if (r == 0)
		{
			if (pLabels[i] == 0)
				confEmpty[0][0]++;//True Positives
			else
				confEmpty[0][1]++;//False posiive
		}
		else
		{
			if (pLabels[i] == 0)
				confEmpty[1][0]++;//False negative
			else
				confEmpty[1][1]++;//True Negatives
		}
//----------------------------------------------------------------------------------

//Bad vs NonBad
//----------------------------------------------------------------------------------
		if (r == 2)
		{
			if (pLabels[i] == 2)
				confBad[0][0]++;//True Positives
			else
				confBad[0][1]++;//False posiive
		}
		else
		{
			if (pLabels[i] == 2)
				confBad[1][0]++;//False negative
			else
				confBad[1][1]++;//True Negatives
		}
//----------------------------------------------------------------------------------

//----------------------------------------------------------------------------------
		
	}

//Good
//----------------------------------------------------------------------------------
	float p_good = confGood[0][0] + confGood[1][0];
	float n_good = confGood[0][1] + confGood[1][1];
	fp_rate_good = confGood[0][1]/ n_good;
	tp_rate_good = confGood[0][0] / p_good;
	precision_good = confGood[0][0] / (confGood[0][0]+ confGood[0][1]);
	accuracy_good = (confGood[0][0] + confGood[1][1]) / (n_good + p_good);
	f_score_good = precision_good*tp_rate_good;
//----------------------------------------------------------------------------------

//Bad
//----------------------------------------------------------------------------------
	float p_bad = confBad[0][0] + confBad[1][0];
	float n_bad = confBad[0][1] + confBad[1][1];
	fp_rate_bad = (float)confBad[0][1] / n_bad;
	tp_rate_bad = (float)confBad[0][0] / p_bad;
	precision_bad = (float)(confBad[0][0]) / (confBad[0][0] + confBad[0][1]);
	accuracy_bad = (float)(confBad[0][0] + confBad[1][1]) / (n_bad + p_bad);
	f_score_bad = precision_bad*tp_rate_bad;
//----------------------------------------------------------------------------------

//Empty
//----------------------------------------------------------------------------------
	float p_empty = confEmpty[0][0] + confEmpty[1][0];
	float n_empty = confEmpty[0][1] + confEmpty[1][1];
	fp_rate_empty = (float)confEmpty[0][1] / n_empty;
	tp_rate_empty = (float)confEmpty[0][0] / p_empty;
	precision_empty = (float)confEmpty[0][0] / (confEmpty[0][0] + confEmpty[0][1]);
	accuracy_empty = (float)(confEmpty[0][0] + confEmpty[1][1]) / (n_empty + p_empty);
	f_score_empty = precision_empty*tp_rate_empty;
//----------------------------------------------------------------------------------

//Final
//----------------------------------------------------------------------------------
	precision_final = (precision_empty+ precision_bad+ precision_good)/3;
	accuracy_final = (accuracy_empty+ accuracy_bad+ accuracy_good)/3;
	f_score_final = (f_score_empty + f_score_bad + f_score_good)/3;
//----------------------------------------------------------------------------------

//Display Results
//----------------------------------------------------------------------------------
	cout << "Class Good:" << endl;
	cout << "Total:" << totalClassGood << endl;
	cout << "Correct:" << classGoodCorrect << endl;
	cout << "Incorrect:" << (totalClassGood - classGoodCorrect) << endl << endl;

	cout << "Class Bad:" << endl;
	cout << "Total:" << totalClassBad << endl;
	cout << "Correct:" << classBadCorrect << endl;
	cout << "Incorrect:" << (totalClassBad - classBadCorrect) << endl << endl;

	cout << "Class Empty:" << endl;
	cout << "Total:" << totalClassEmpty << endl;
	cout << "Correct:" << classEmptyCorrect << endl;
	cout << "Incorrect:" << (totalClassEmpty - classEmptyCorrect) << endl << endl;

	cout << endl << "Good:" << endl;
	cout << "Precision: " << precision_good *100 << "%" << endl;
	cout << "Accuracy: " << accuracy_good *100 << "%" << endl;
	cout << "F-score: " << f_score_good *100 << "%" << endl;

	cout << endl << "Bad:" << endl;
	cout << "Precision: " << precision_bad * 100 << "%" << endl;
	cout << "Accuracy: " << accuracy_bad * 100 << "%" << endl;
	cout << "F-score: " << f_score_bad * 100 << "%" << endl;

	cout << endl << "Empty:" << endl;
	cout << "Precision: " << precision_empty * 100 << "%" << endl;
	cout << "Accuracy: " << accuracy_empty * 100 << "%" << endl;
	cout << "F-score: " << f_score_empty * 100 << "%" << endl;

	cout << endl << "Final:" << endl;
	cout << "Precision: " << precision_final * 100 << "%" << endl;
	cout << "Accuracy: " << accuracy_final * 100 << "%" << endl;
	cout << "F-score: " << f_score_final * 100 << "%" << endl;
//----------------------------------------------------------------------------------

	if (!filename_to_save.empty())
	{
		model->save(filename_to_save);
	}
}