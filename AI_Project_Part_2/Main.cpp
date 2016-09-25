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
#define TRAINING_SIZE 108
#define TEST_SIZE 52

using namespace std;
using namespace cv;
using namespace ml;

//Feature Extraction Methods
//--------------------------------------
void extractFeatures(string *,boolean,const char *,boolean);
void calculateGLCM(int, int, int, int);
float maxProb();
float energy();
float homogeneity();
float contrast();
float entropy();
float correlation();
float meani();
float meanj();
float deviationi();
float deviationj();
//--------------------------------------

//Neural Network Methods
//--------------------------------------
void ExtractData(Part *,boolean ,const char*,boolean);
void setUpMatrices(Part *,boolean , int *,float [][48]);
inline TermCriteria TC(int, double);
static void predict_display_results(const Ptr<StatModel>&, const string& filename_to_save);
static bool train_mlp_classifier(const string& filename_to_save);
//--------------------------------------

//Feature Extraction Variables
//--------------------------------------
const int MAX_SIZE = 16;										//16-tone grayscale images are used
const int IMAGE_SIZE_WIDTH_MAX = 30;							//Maximum size from the images in the dataset
const int IMAGE_SIZE_HEIGHT_MAX = 250;

int x[IMAGE_SIZE_HEIGHT_MAX][IMAGE_SIZE_WIDTH_MAX] = { 0 };		//Stores the pixel values of the image
int GLCM[MAX_SIZE][MAX_SIZE] = { 0 };							//corresponding GLCM matrix
float p[MAX_SIZE][MAX_SIZE] = { 0 };							//Normalized Matrix

string TrainingImagesArray[TRAINING_SIZE];
string TestingImagesArray[TEST_SIZE];

//--------------------------------------

//Neural Network Variables
//--------------------------------------
Part training[TRAINING_SIZE];
Part test[TEST_SIZE];

//Training Purposes
int labels[TRAINING_SIZE];
float trainingData[TRAINING_SIZE][48];

//Prediction Purposes
int pLabels[TEST_SIZE];
float pData[TEST_SIZE][48];

//Confusion Matrix
int confGood[2][2];
int confBad[2][2];
int confEmpty[2][2];
//--------------------------------------

int main(int argc, char * argv[]) {

	const char * path = argv[1];

	boolean console = false;

	cout << "Extracting Features for Training Data..." << endl;
	extractFeatures(TrainingImagesArray,1,path, console);							//Extract Training Features

	cout << "Extracting Features for Test Data..." << endl;
	extractFeatures(TestingImagesArray,0,path, console);							//Extract Testing Features

	cout << "Extract Training Features from textfile..." << endl;
	ExtractData(training,1,path, console);											//Extract Data from train textfile
	cout << "Store Training Data in Matrices..." << endl;	
	setUpMatrices(training,1, labels, trainingData);						//Split data into appropriate training matrices

	cout << "Extract Test Features from textfile..." << endl;
	ExtractData(test, 0,path, console);											//Extract Data from test textfile
	cout << "Store Test Data in Matrices..." << endl;
	setUpMatrices(test, 0, pLabels, pData);									//Split data into appropriate test matrices

	cout << "Begin Training..." << endl;
	train_mlp_classifier("classifier.txt");									//Initialize and train neural network
	system("pause");

}

//Neural Network Methods
//--------------------------------------
void ExtractData(Part *p,boolean t,const char *path,boolean console)
{
	std::ifstream file;
	string line;

	if (console)
	{
		string temp1;
		if (t)
			temp1 = "train.txt";
		else
			temp1 = "test.txt";

		string temp2(path);
		string temp3 = temp2 + temp1;
		file.open(temp3);
	}
	else
	{
		if (t)
			file.open("C:/Users/ahmadbk/Desktop/AI_Project_Part_2/AI_Project_Part_2/train.txt");
		else
			file.open("C:/Users/ahmadbk/Desktop/AI_Project_Part_2/AI_Project_Part_2/test.txt");
	}

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

void setUpMatrices(Part *p,boolean t,int *tempLabels,float tempTrainingData[][48])
{
	int size = 0;
	if (t)
		size = TRAINING_SIZE;
	else
		size = TEST_SIZE;

	for (int i = 0; i < size; i++)
		tempLabels[i] = p[i].getResult();
	

	for (int i = 0; i < size; i++)
	{
		int k = 0;
		for (int j = 0; j < 8; j++)
		{
			tempTrainingData[i][k++] = p[i].partData[j].f.getContrast();
			tempTrainingData[i][k++] = p[i].partData[j].f.getCorrelation();
			tempTrainingData[i][k++] = p[i].partData[j].f.getEnergy();
			tempTrainingData[i][k++] = p[i].partData[j].f.getEntropy();
			tempTrainingData[i][k++] = p[i].partData[j].f.getHomogeneity();
			tempTrainingData[i][k++] = p[i].partData[j].f.getMaxProb();
		}
	}
}

static bool train_mlp_classifier(const string& filename_to_save)
{
	double duration;
	std::clock_t start = clock();

	const int class_count = 3;	
	Mat train_data = Mat(TRAINING_SIZE, 48, CV_32FC1, &trainingData);
	Mat responses = Mat(TRAINING_SIZE, 1, CV_32S, &labels);

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
	//int layer_sz[] = { train_data.cols,10,class_count };
	int layer_sz[] = { train_data.cols,class_count };
	int nlayers = (int)(sizeof(layer_sz) / sizeof(layer_sz[0]));
	Mat layer_sizes(1, nlayers, CV_32S, layer_sz);

#if 1
	int method = ANN_MLP::BACKPROP;
	double method_param = 0.000001;
	int max_iter = 10000;
#else
	int method = ANN_MLP::RPROP;
	double method_param = 0.1;
	int max_iter = 1000;
#endif

	Ptr<TrainData> tdata = TrainData::create(train_data, ROW_SAMPLE, train_responses);

	cout << "Training the classifier..." << endl;
	model = ANN_MLP::create();
	model->setLayerSizes(layer_sizes);
	model->setActivationFunction(ANN_MLP::SIGMOID_SYM, 0, 0);
	model->setTermCriteria(TC(max_iter, 0));
	model->setTrainMethod(method, method_param);
	model->train(tdata);
	//cout << endl;

	duration = (clock() - start) / CLOCKS_PER_SEC;
	cout << "Total Training Time for " << TRAINING_SIZE << " images is: " << duration << " sec" << endl;

	cout << "Begin Prediction..." << endl;
	cout << "Neural Network Results..." << endl << endl;
	predict_display_results(model, filename_to_save);

	return true;
}

inline TermCriteria TC(int iters, double eps)
{
	return TermCriteria(TermCriteria::MAX_ITER + (eps > 0 ? TermCriteria::EPS : 0), iters, eps);
}

static void predict_display_results(const Ptr<StatModel>& model, const string& filename_to_save)
{
	Mat pdata = Mat(TEST_SIZE, 48, CV_32FC1, &pData);   //Loading the rest of the data for prediction
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
	fp_rate_good = (float)confGood[0][1]/ n_good;
	tp_rate_good = (float)confGood[0][0] / p_good;
	precision_good = (float)confGood[0][0] / (confGood[0][0]+ confGood[0][1]);
	accuracy_good = (float)(confGood[0][0] + confGood[1][1]) / (n_good + p_good);
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
//--------------------------------------

//Feature Extraction Methods
//--------------------------------------
void extractFeatures(string *imagesArray,boolean t, const char * path,boolean console)
{
	int size = 0;
	if (t)
		size = TRAINING_SIZE;
	else
		size = TEST_SIZE;

	//double duration;
	//std::clock_t start;
	
	char* imageFilePath;
	char* temp4;
	if (console)
	{
		string temp1;
		if (t)
			temp1 = "train/*";
		else
			temp1 = "test/*";
		
		string temp2(path);
		string temp3 = temp2 + temp1;
		temp4 = new char[temp3.length() + 1];
		string temp5(temp4);
		strcpy_s(temp4, temp5.length(), temp3.c_str());
		imageFilePath = temp4;
		delete temp4;
	}
	else
	{
		if (t)
			imageFilePath = "C:/Users/ahmadbk/Desktop/AI_Project_Part_2/AI_Project_Part_2/train/*";	//path to all the train images
		else
			imageFilePath = "C:/Users/ahmadbk/Desktop/AI_Project_Part_2/AI_Project_Part_2/test/*";	//path to all the test images
	}

	string temp(imageFilePath);
	string imagePath = temp.substr(0, (temp.length() - 2));

	//Obtain the names of all the images in the filepath
	//------------------------------------------------------------------------------
	WIN32_FIND_DATA search_data;
	memset(&search_data, 0, sizeof(WIN32_FIND_DATA));
	HANDLE handle = FindFirstFile(imageFilePath, &search_data);
	int count = 0;
	while (handle != INVALID_HANDLE_VALUE)
	{
		if (count > 1)
			imagesArray[count - 2] = search_data.cFileName;
		count++;
		if (FindNextFile(handle, &search_data) == FALSE)
			break;
	}
	FindClose(handle);
	//------------------------------------------------------------------------------

	int orientation[4] = { 0,45,90,135 };
	int distances[2] = { 1,3 };
	ofstream result;

	if(t)
		result.open("train.txt");
	else
		result.open("test.txt");

	cv::Mat imgOriginal;		// input image
	cv::Mat imgGrayscale;		// grayscale of input image
	cv::Mat finalImage;			//16-tone grayscae image

	//result << "ImageName;Distance#Orientation#MaxProbabilty#Energy#Homogeneity#Contrast#Correlation#Entropy;Distance#Orientation...\n";

	//Run the program for all the images found in the folder
	//------------------------------------------------------------------------------
	for (int w = 0; w < size; w++)
	{
		string imgName = imagesArray[w];

		result << imgName << ";";

		string imagePathl;

		if (console)
		{
			string temp1;
			if (t)
				temp1 = "train/";
			else
				temp1 = "test/";

			string temp(path);
			string temp3 = temp + temp1;
			//imagePath = temp.substr(0, (temp.length() - 2));
			imagePath = temp3;
		}
		else
		{
			if (t)
				imagePath = "C:/Users/ahmadbk/Desktop/AI_Project_Part_2/AI_Project_Part_2/train/";
			else
				imagePath = "C:/Users/ahmadbk/Desktop/AI_Project_Part_2/AI_Project_Part_2/test/";
		}

		string path = imagePath + imgName;
		imgOriginal = cv::imread(path);				// open image

		if (imgOriginal.empty()) {									// if unable to open image
			std::cout << "error: image not read from file\n\n";		// show error message on command line
																	//return(0);												// and exit program
		}

		cv::cvtColor(imgOriginal, imgGrayscale, CV_BGR2GRAY);		// convert to grayscale

																	//Convert the grayscale image to 16-tone grayscal using lookup table
																	//------------------------------------------------------------------------------
		uchar *p;
		cv::Mat lookuptable(1, 256, CV_8U);
		p = lookuptable.data;
		for (int i = 0; i < 256; ++i)
			p[i] = (i / 16);
		LUT(imgGrayscale, lookuptable, finalImage);

		for (int i = 0; i < finalImage.rows; i++)
		{
			for (int j = 0; j < finalImage.cols; j++)
			{
				int pixel = (int)(finalImage.at<uchar>(i, j));
				x[i][j] = pixel;
			}
		}
		//------------------------------------------------------------------------------

		//Calculate all features and store in textfile
		//------------------------------------------------------------------------------
		for (int i = 0; i < 2; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				result << distances[i] << "#";
				//cout << "Distance: " << distances[i] << endl;
				result << orientation[j] << "#";
				//cout << "Orientation: " << orientation[j] << endl;

				calculateGLCM(orientation[j], distances[i], finalImage.cols, finalImage.rows);

				float t1 = maxProb();
				//cout << "Max Probability: " << t1 << endl;
				result << t1 << "#";

				float t2 = energy();
				//cout << "Energy: " << t2 << endl;
				result << t2 << "#";

				float t3 = homogeneity();
				//cout << "Homogeneity: " << t3 << endl;
				result << t3 << "#";

				float t4 = contrast();
				//cout << "Contrast: " << t4 << endl;
				result << t4 << "#";

				float t5 = correlation();
				//cout << "Correlation: " << t5 << endl;
				result << t5 << "#";

				float t6 = entropy();
				//cout << "Entropy: " << t6 << endl;
				result << t6;

				if (i != 1 || j != 3)
					result << ";";
				else
					result << "\n";
				//cout << endl;

			}
		}
		//------------------------------------------------------------------------------
	}
	//------------------------------------------------------------------------------

	result.close();
}

void calculateGLCM(int angle, int distance, int cols, int rows)
{
	for (int i = 0; i < MAX_SIZE; i++)
	{
		for (int j = 0; j < MAX_SIZE; j++)
		{
			int temp = 0;
			for (int k = 0; k < rows; k++)
			{
				for (int l = 0; l < cols; l++)
				{
					if (x[k][l] == i)
					{
						if (angle == 0)
						{
							if (l + distance < cols)
								if (x[k][l + distance] == j)
									temp += 1;

							if (l - distance >= 0)
								if (x[k][l - distance] == j)
									temp += 1;
						}
						if (angle == 45)
						{
							if (l + distance < cols && k - distance >= 0)
								if (x[k - distance][l + distance] == j)
									temp += 1;

							if (l - distance >= 0 && k + distance < rows)
								if (x[k + distance][l - distance] == j)
									temp += 1;
						}

						if (angle == 90)
						{
							if (k - distance >= 0)
								if (x[k - distance][l] == j)
									temp += 1;

							if (k + distance < rows)
								if (x[k + distance][l] == j)
									temp += 1;
						}

						if (angle == 135)
						{
							if (k - distance >= 0 && l - distance >= 0)
								if (x[k - distance][l - distance] == j)
									temp += 1;

							if (k + distance < rows && l + distance < cols)
								if (x[k + distance][l + distance] == j)
									temp += 1;
						}
					}
				}
			}
			GLCM[i][j] = temp;
		}
	}

	float total = 0;
	for (int i = 0; i < MAX_SIZE; i++)
	{
		for (int j = 0; j < MAX_SIZE; j++)
		{
			total += GLCM[i][j];
		}
	}

	for (int i = 0; i < MAX_SIZE; i++)
	{
		for (int j = 0; j < MAX_SIZE; j++)
		{
			p[i][j] = (float)(GLCM[i][j] / total);
		}
	}
}

float maxProb()
{
	float temp = 0;
	for (int i = 0; i < MAX_SIZE; i++)
	{
		for (int j = 0; j < MAX_SIZE; j++)
		{
			if (p[i][j] > temp)
				temp = p[i][j];
		}
	}
	return temp;
}

float energy()
{
	float total = 0;
	for (int i = 0; i < MAX_SIZE; i++)
	{
		for (int j = 0; j < MAX_SIZE; j++)
		{
			total += p[i][j] * p[i][j];
		}
	}
	return total;
}

float homogeneity()
{
	float temp = 0;
	for (int i = 0; i < MAX_SIZE; i++)
	{
		for (int j = 0; j < MAX_SIZE; j++)
		{
			temp += p[i][j] / (1 + abs(i - j));
		}
	}
	return temp;
}

float contrast()
{
	float temp = 0;
	for (int i = 0; i < MAX_SIZE; i++)
	{
		for (int j = 0; j < MAX_SIZE; j++)
		{
			temp += p[i][j] * (abs(i - j))*(abs(i - j));
		}
	}
	return temp;
}

float entropy()
{
	float temp = 0;
	for (int i = 0; i < MAX_SIZE; i++)
	{
		for (int j = 0; j < MAX_SIZE; j++)
		{
			if (p[i][j] != 0)
				temp += p[i][j] * (-log(p[i][j]));
		}
	}
	return temp;
}

float correlation()
{
	float mi = meani();
	float mj = meanj();
	float di = deviationi();
	float dj = deviationj();
	float total = 0;
	for (int i = 0; i < MAX_SIZE; i++)
	{
		for (int j = 0; j < MAX_SIZE; j++)
		{
			float temp1 = i - mi;
			float temp2 = j - mj;
			float temp3 = di*dj;
			total += (p[i][j] * temp1 * temp2) / temp3;
		}
	}
	return total;
}

float meani()
{
	float total = 0;
	for (int i = 0; i < MAX_SIZE; i++)
	{
		for (int j = 0; j < MAX_SIZE; j++)
		{
			total += i*p[i][j];
		}
	}
	return total;
}

float meanj()
{
	float total = 0;
	for (int i = 0; i < MAX_SIZE; i++)
	{
		for (int j = 0; j < MAX_SIZE; j++)
		{
			total += j*p[i][j];
		}
	}
	return total;
}

float deviationi()
{
	float mi = meani();
	float total = 0;
	for (int i = 0; i < MAX_SIZE; i++)
	{
		for (int j = 0; j < MAX_SIZE; j++)
		{
			float temp = (i - mi)*(i - mi);
			total += p[i][j] * temp;
		}
	}
	return sqrt(total);
}

float deviationj()
{
	float mj = meanj();
	float total = 0;
	for (int i = 0; i < MAX_SIZE; i++)
	{
		for (int j = 0; j < MAX_SIZE; j++)
		{
			float temp = (j - mj)*(j - mj);
			total += p[i][j] * temp;
		}
	}
	return sqrt(total);
}
//--------------------------------------
