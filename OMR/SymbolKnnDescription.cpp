#include "SymbolKnnDescription.h"
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\ml\ml.hpp>
#include <opencv2\opencv.hpp>

#include <map>
#include <iostream>
#include <filesystem>

using namespace cv;
using namespace std;
using namespace std::experimental::filesystem;

struct SymbolKnnDescription::impl {
	impl();
	~impl();

	CvKNearest symbolKnn;
	CvKNearest noteKnn;

	map<string, int> GetTrainData(char* path);
	void TrainKnn(map<string, int> input, int knnIndex);
};
SymbolKnnDescription::impl::impl() {}
SymbolKnnDescription::impl::~impl() {}

map<string, int> SymbolKnnDescription::impl::GetTrainData(char * path)
{
	map<string, int> result;
	int index = -1;
	for (directory_entry dirEntry : recursive_directory_iterator(path)) {
		if (dirEntry.path().extension().string() == "") index++;
		else if (dirEntry.path().extension().string() == ".png")
			result.insert(pair<string, int>(dirEntry.path().string(), index));
		else throw "invalid trainning data !";
	}

	return result;
}

void SymbolKnnDescription::impl::TrainKnn(map<string, int> input, int knnIndex)
{
	int trainingSetSize = input.size();
	CvMat *trainingSet = cvCreateMat(trainingSetSize, 400, CV_32FC1);
	CvMat *labels = cvCreateMat(trainingSetSize, 1, CV_32FC1);

	int i = 0;
	for (map<string, int>::iterator it = input.begin(); it != input.end(); ++it) {
		Mat inputData = imread(it->first, CV_LOAD_IMAGE_GRAYSCALE);
		Mat trainData(20, 20, CV_32FC1);
		resize(inputData, trainData, trainData.size(), CV_INTER_AREA);
		//將map的index值放進label mat內
		labels->data.fl[i] = it->second;

		//將從map拿到的圖片，轉為20x20後放入trainset內
		for (int j = 0; j < 400; j++) {
			trainingSet->data.fl[i * 400 + j] = trainData.data[j];
		}
		i++;
	}

	if (knnIndex == 1) symbolKnn.train(trainingSet, labels);
	else if (knnIndex == 2) noteKnn.train(trainingSet, labels);
	else throw "invalid input !";

	cvReleaseMat(&trainingSet);
	cvReleaseMat(&labels);
}

SymbolKnnDescription::SymbolKnnDescription(char* symbolPath, char* notePath) :pimpl(new impl)
{
	map<string, int> symbolTrainData = pimpl->GetTrainData(symbolPath);
	map<string, int> noteTrainData = pimpl->GetTrainData(notePath);

	pimpl->TrainKnn(symbolTrainData, 1);
	pimpl->TrainKnn(noteTrainData, 2);

	//for (map<string, int>::iterator it = pimpl->symbolTrainData.begin(); it != pimpl->symbolTrainData.end(); ++it)
	//	std::cout << "path = " << it->first << " ,index = " << it->second << '\n';

	system("PAUSE");
}

SymbolKnnDescription::~SymbolKnnDescription()
{
}

float SymbolKnnDescription::FindNearest(Mat sample, int noteType)
{
	int K = 5;
	Mat sampleNormalize(20, 20, CV_32FC1);
	resize(sample, sampleNormalize, sampleNormalize.size(), CV_INTER_AREA);
	float _sampleData[400];
	Mat sampleData(1, 400, CV_32FC1, _sampleData);
	memcpy(sampleData.data, sampleNormalize.data, 400);
	Mat distance(1, K, CV_32FC1);

	float result = 0.0f;
	switch (noteType)
	{
	case(1):
		result = pimpl->symbolKnn.find_nearest(sampleData, K, 0, 0, 0, &distance);
		break;

	case(2):
		result = pimpl->noteKnn.find_nearest(sampleData, K, 0, 0, 0, &distance);
		break;
	default:
		return -1.0f;
	}

	return result;
}
