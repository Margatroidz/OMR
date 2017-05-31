#pragma once
#include <memory>
#include <opencv2\core\core.hpp>

class SymbolKnnDescription {
public:
	SymbolKnnDescription(char* symbolPath, char* notePath);
	~SymbolKnnDescription();
	//noteType = 1, 表示尋找非音符的其他符號，noteTpye = 2, 表示尋找音符
	float FindNearest(cv::Mat sample, int noteType);
private:
	struct impl;
	std::unique_ptr<impl> pimpl;
};