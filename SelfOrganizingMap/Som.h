#include <vector>
#include <cmath>
#include <ctime>
#include <iostream>
#include <algorithm>
#include <random>
#include "InputDataManager.h"
#include <map>
#include <fstream>

#pragma once
class Som
{
private:
    std::vector<std::vector<std::vector<double>>> neuronGrid;
    std::vector<std::vector<int>> labelMap;
    std::vector<std::vector<double>> features;
    std::vector<int> labels;

    double learningRate;
    double radius;
    int gridRow;
    int gridCol;
    InputDataManager inputDataManager;

    void initGrid() {
        srand(static_cast<unsigned int>(time(0)));
        int featureSize = features[0].size();

        neuronGrid.resize(gridRow, std::vector<std::vector<double>>(gridCol, std::vector<double>(featureSize)));

        for (int i = 0; i < gridRow; i++) {
            for (int j = 0; j < gridCol; j++) {
                for (int k = 0; k < featureSize; k++) {
                    neuronGrid[i][j][k] = features[rand() % features.size()][rand() % featureSize];
                }
            }
        }

        labelMap.resize(gridRow, std::vector<int>(gridCol, -1));
    }

    double euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); i++) {
            sum += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return std::sqrt(sum);
    }

    double manhattanDistance(const std::pair<int, int> a, const std::pair<int, int> b) {
        return std::abs(a.first - b.first) + std::abs(a.second - b.second);
    }

    double neighborhoodFunction(const std::pair<int, int> a, const std::pair<int, int> b) {
        double mDist = manhattanDistance(a, b);
        return std::exp(-(std::pow(mDist, 2) / (2 * std::pow(radius, 2))));
    }

    std::pair<int, int> findBmu(const std::vector<double>& input) {
        int x = -1;
        int y = -1;
        double min = std::numeric_limits<double>::max();

        for (size_t i = 0; i < gridRow; i++) {
            for (size_t j = 0; j < gridCol; j++) {
                double distance = euclideanDistance(neuronGrid[i][j], input);
                if (distance < min) {
                    x = i;
                    y = j;
                    min = distance;
                }
            }
        }
        return std::make_pair(x, y);
    }

    void updateWeights(std::pair<int, int> bmuIndex, const std::vector<double>& input) {
        for (size_t i = 0; i < gridRow; i++) {
            for (size_t j = 0; j < gridCol; j++) {
                for (size_t z = 0; z < input.size(); z++) {
                    double weightUpdate = learningRate *
                        neighborhoodFunction(bmuIndex, std::make_pair(i, j)) *
                        (input[z] - neuronGrid[i][j][z]);

                    neuronGrid[i][j][z] += weightUpdate;
                }
            }
        }
    }

    double calculateAccuracy(const std::vector<std::vector<double>>& X_test, const std::vector<int>& Y_test) {
        int correct = 0;
        for (size_t i = 0; i < X_test.size(); i++) {
            int predictedLabel = predict(X_test[i]);
            if (predictedLabel == Y_test[i]) {
                correct++;
            }
        }
        return (static_cast<double>(correct) / Y_test.size()) * 100;
    }

    void updateLabelMap(std::pair<int, int> bmuIndex, int label) {
        labelMap[bmuIndex.first][bmuIndex.second] = label;
    }

public:
    Som(std::vector<std::vector<double>> features, std::vector<int> labels, int gridRow, int gridCol, double learningRate, double radius) :
        features(features),
        labels(labels),
        gridRow(gridRow),
        gridCol(gridCol),
        learningRate(learningRate),
        radius(radius) {
        initGrid();
        inputDataManager = InputDataManager();
        auto pair = inputDataManager.shuffle(features, labels);
        this->features = pair.first;
        this->labels = pair.second;
    }

    void train(int epochs, float train_test_split, std::string outputFilePath) {
        std::ofstream file(outputFilePath);
        if (!file.is_open()) {
            std::cerr << "Can't open the output file path" << std::endl;
            return;
        }

        DataTrainTestSplit split = inputDataManager.train_test_split(features, labels, train_test_split);

        for (size_t i = 0; i < epochs; i++) {
            for (size_t row = 0; row < split.X_train.size(); row++) {
                std::vector<double> input = split.X_train[row];
                int label = split.Y_train[row];

                std::pair<int, int> bmuIndex = findBmu(input);
                updateWeights(bmuIndex, input);
                updateLabelMap(bmuIndex, label);
            }
            double accuracy = calculateAccuracy(split.X_test, split.Y_test);
            int epoch = i + 1;
            file << epoch << "," << accuracy << std::endl;
            learningRate *= 0.995;
            radius *= 0.995;
            std::cout << "Epoch: " << epoch << ", Learning Rate: " << learningRate << ", Radius: " << radius << ", Accuracy: " << accuracy << std::endl;
        }
        file.close();
    }

    int predict(const std::vector<double>& input) {
        std::pair<int, int> bmuIndex = findBmu(input);
        return labelMap[bmuIndex.first][bmuIndex.second];
    }
};
