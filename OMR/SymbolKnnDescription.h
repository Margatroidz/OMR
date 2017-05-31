#pragma once
#include <memory>
#include <opencv2\core\core.hpp>

class SymbolKnnDescription {
public:
	SymbolKnnDescription(char* symbolPath, char* notePath);
	~SymbolKnnDescription();
	//noteType = 1, ��ܴM��D���Ū���L�Ÿ��AnoteTpye = 2, ��ܴM�䭵��
	float FindNearest(cv::Mat sample, int noteType);
private:
	struct impl;
	std::unique_ptr<impl> pimpl;
};