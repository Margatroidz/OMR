#pragma once
#include <memory>
#include <opencv2\core\core.hpp>

class OMRKnnDescription {
public:
	OMRKnnDescription(char* symbolPath, char* notePath);
	~OMRKnnDescription();
	//sample為輸入的影像，大小不限，noteType = 1, 表示尋找非音符的其他符號，noteTpye = 2, 表示尋找音符
	float FindNearestSymbol(cv::Mat sample);
	float FindNearestNoteElement(cv::Mat sample);
	std::string GetSymbolName(float index);
	std::string GetNoteElementName(float index);
private:
	struct impl;
	std::unique_ptr<impl> pimpl;
};