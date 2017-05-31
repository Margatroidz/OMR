#include "OMRKnnDescription.h"
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\ml\ml.hpp>
#include <opencv2\opencv.hpp>

#include <map>
#include <iomanip>
#include <iostream>
#include <filesystem>

using namespace cv;
using namespace std;
using namespace std::experimental::filesystem;

struct OMRKnnDescription::impl {
	impl();
	~impl();

	CvKNearest symbolKnn;
	CvKNearest noteElementKnn;
	vector<string> indexToSymbolName;
	vector<string> indexToNoteElementName;

	map<string, int> GetTrainData(char* path, vector<string>& correspondentName);
	void TrainKnn(map<string, int> input, CvKNearest& knnDataSet);
	float FindNearestSymbol(Mat sample, CvKNearest knn);
};
OMRKnnDescription::impl::impl() {}
OMRKnnDescription::impl::~impl() {}

map<string, int> OMRKnnDescription::impl::GetTrainData(char * path, vector<string>& correspondentName)
{
	map<string, int> result;
	int index = -1;
	for (directory_entry dirEntry : recursive_directory_iterator(path)) {
		if (dirEntry.path().extension().string() == "") {
			correspondentName.push_back(dirEntry.path().filename().string());
			std::cout << dirEntry.path().filename().string() << std::endl;
			index++;
		}
		else if (dirEntry.path().extension().string() == ".png")
			result.insert(pair<string, int>(dirEntry.path().string(), index));
		else throw "invalid trainning data !";
	}

	return result;
}

void OMRKnnDescription::impl::TrainKnn(map<string, int> input, CvKNearest& knnDataSet)
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

	knnDataSet.train(trainingSet, labels);
	cvReleaseMat(&trainingSet);
	cvReleaseMat(&labels);
}

float OMRKnnDescription::impl::FindNearestSymbol(Mat sample, CvKNearest knn)
{
	int K = 5;
	Mat sampleNormalize(20, 20, CV_32FC1);
	resize(sample, sampleNormalize, sampleNormalize.size(), CV_INTER_AREA);
	CvMat *sampleData = cvCreateMat(1, 400, CV_32FC1);
	for (int i = 0; i < 400; i++) sampleData->data.fl[i] = sampleNormalize.data[i];
	CvMat *distance = cvCreateMat(1, K, CV_32FC1);

	float result = 0.0f;
	result = knn.find_nearest(sampleData, K, 0, 0, 0, distance);

	for (int i = 0; i < K; i++) cout << "No\. " << i << " distance = " << setprecision(3)<< (float)distance->data.fl[i] << endl;

	cvReleaseMat(&distance);
	cvReleaseMat(&sampleData);
	return result;
}

OMRKnnDescription::OMRKnnDescription(char* symbolPath, char* notePath) :pimpl(new impl)
{
	map<string, int> symbolTrainData = pimpl->GetTrainData(symbolPath, pimpl->indexToSymbolName);
	map<string, int> noteTrainData = pimpl->GetTrainData(notePath, pimpl->indexToNoteElementName);

	pimpl->TrainKnn(symbolTrainData, pimpl->symbolKnn);
	pimpl->TrainKnn(noteTrainData, pimpl->noteElementKnn);

	//for (map<string, int>::iterator it = pimpl->symbolTrainData.begin(); it != pimpl->symbolTrainData.end(); ++it)
	//	std::cout << "path = " << it->first << " ,index = " << it->second << '\n';
}

OMRKnnDescription::~OMRKnnDescription()
{
}

float OMRKnnDescription::FindNearestSymbol(Mat sample)
{
	return pimpl->FindNearestSymbol(sample, pimpl->symbolKnn);
}

float OMRKnnDescription::FindNearestNoteElement(Mat sample)
{
	return pimpl->FindNearestSymbol(sample, pimpl->noteElementKnn);
}

string OMRKnnDescription::GetSymbolName(float index)
{
	return pimpl->indexToSymbolName[(int)index];
}

string OMRKnnDescription::GetNoteElementName(float index)
{
	return pimpl->indexToNoteElementName[(int)index];
}
