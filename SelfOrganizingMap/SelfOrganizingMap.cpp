#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "SOM.h"


std::vector<std::string> split(std::string s, std::string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        token = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back(token);
    }

    res.push_back(s.substr(pos_start));
    return res;
}

int convertLabelToInt(const std::string& label) {
    if (label == "Iris-setosa")
        return 0;
    else if (label == "Iris-versicolor")
        return 1;
    return 2;
}

int main() {
    std::ifstream file("bezdekIris.data");

    if (!file) {
        std::cerr << "Unable to open file" << std::endl;
        return 1;
    }

    std::string line;
    std::vector<std::vector<double>> features;
    std::vector<int> labels;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::vector<std::string> arr = split(line, ",");

        std::vector<double> row_features;
        for (size_t i = 0; i < arr.size(); i++) {
            if (i == arr.size() - 1) {
                labels.push_back(convertLabelToInt(arr[i]));
            }
            else {
                row_features.push_back(std::stod(arr[i]));
            }
        }

        features.push_back(row_features);
    }
    file.close();
    /*std::cout << "Features" << std::endl;
    for (auto& i : features) {
        for (auto& j : i) {
            std::cout << j << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << "Labels" << std::endl;
    for (auto& i : labels) {
        std::cout << i << std::endl;
    }*/

    Som som = Som(features, labels, 8, 8, 0.21, 1);
    som.train(60, 0.35, "./accuracy.txt");
    return 0;
}