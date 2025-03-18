#include <vector>

#pragma once
class DataTrainTestSplit
{
public:
	std::vector<std::vector<double>> X_train;
	std::vector<std::vector<double>> X_test;

	std::vector<int> Y_train;
	std::vector<int> Y_test;
};

