#pragma once
#include <memory>

class SymbolKnnDescription {
public:
	SymbolKnnDescription(char* symbolPath, char* notePath);
	~SymbolKnnDescription();
private:
	struct impl;
	std::unique_ptr<impl> pimpl;
};