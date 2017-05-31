#pragma once
#include <memory>
#include <opencv2\core\core.hpp>

class OMRKnnDescription {
public:
	OMRKnnDescription(char* symbolPath, char* notePath);
	~OMRKnnDescription();
	//sample����J���v���A�j�p�����AnoteType = 1, ��ܴM��D���Ū���L�Ÿ��AnoteTpye = 2, ��ܴM�䭵��
	float FindNearest(cv::Mat sample, int noteType);
	std::string GetSymbolName(float index);
	std::string GetNoteName(float index);
private:
	struct impl;
	std::unique_ptr<impl> pimpl;
};