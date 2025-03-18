#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <numeric>

#include "DataTrainTestSplit.h"

#pragma once
class InputDataManager
{
public:
	static std::pair<std::vector<std::vector<double>>, std::vector<int>>
		shuffle(std::vector<std::vector<double>> v1, std::vector<int> v2) {
		if (v1.size() != v2.size()) {
			std::cerr << "vector sizes must be the same" << std::endl;
			return std::make_pair(std::vector<std::vector<double>>(), std::vector<int>());
		}

		std::vector<size_t> indices(v1.size());
		std::iota(indices.begin(), indices.end(), 0);

		auto rng = std::default_random_engine{ std::mt19937{ std::random_device{}()} };
		std::shuffle(indices.begin(), indices.end(), rng);

		std::vector<std::vector<double>> shuffledV1(v1.size());
		std::vector<int> shuffledV2(v2.size());

		for (size_t i = 0; i < indices.size(); i++) {
			shuffledV1[i] = v1[indices[i]];
			shuffledV2[i] = v2[indices[i]];
		}
		return std::make_pair(shuffledV1, shuffledV2);
	}

	static DataTrainTestSplit train_test_split(std::vector<std::vector<double>> features, std::vector<int> labels, float trainTestSplit) {
		if (trainTestSplit < 0 || trainTestSplit > 1) {
			std::cerr << "train test split must be between 0 and 1" << std::endl;
			return DataTrainTestSplit();
		}
		DataTrainTestSplit split = DataTrainTestSplit();
		size_t testIndexBegin = std::round(trainTestSplit * features.size());

		for (size_t i = 0; i < testIndexBegin; i++) {
			split.X_test.push_back(features[i]);
			split.Y_test.push_back(labels[i]);
		}
		for (size_t i = testIndexBegin; i < features.size(); i++) {
			split.X_train.push_back(features[i]);
			split.Y_train.push_back(labels[i]);
		}

		return split;
	}
};

