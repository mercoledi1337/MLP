#include <filesystem>

#include "MLP.h"
#include <iostream>
#include <numeric>
#include <random>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

struct TrainingData {
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> targets;
};

void shuffleData(std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& targets) {

    std::vector<size_t> indicas(inputs.size());
    std::iota(indicas.begin(), indicas.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(indicas.begin(), indicas.end(), g);

    std::vector<std::vector<float>> shuffledInputs;
    std::vector<std::vector<float>> shuffledTargets;
    shuffledInputs.reserve(indicas.size());
    shuffledTargets.reserve(targets.size());

    for (size_t i : indicas) {
        shuffledInputs.push_back(inputs[i]);
        shuffledTargets.push_back(targets[i]);
    }

    inputs = std::move(shuffledInputs);
    targets = std::move(shuffledTargets);
}

TrainingData loadPhotos(std::string folderPath, int numClasses) {
    TrainingData td;

    for (int i = 0 ; i < numClasses ; ++i) {
        std::string classPath = folderPath + "/" + std::to_string(i);
        std::cout << "Reading from directory: " << classPath << std::endl;

        for (const auto& entry : std::filesystem::directory_iterator(classPath)) {
            if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {

                cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
                if (img.empty()) continue;

                cv::Mat resized;
                cv::resize(img, resized, cv::Size(64, 64));
                std::vector<float> inputVec;
                inputVec.reserve(4096);
                for (int r = 0; r < 64; ++r) {
                    for (int c = 0; c < 64; ++c) {
                        inputVec.push_back(resized.at<uchar>(r, c) / 255.0);
                    }
                }

                std::vector<float> target(numClasses, 0.0);
                target[i] = 1.0;

                td.inputs.push_back(inputVec);
                td.targets.push_back(target);
            }
        }
    }
    return td;
}

int main() {

    // std::string path = "../../data";
    //
    // TrainingData data = loadPhotos("../../data", 3);

    // if(data.inputs.empty()) {
    //     std::cout << "Photos not found!" << std::endl;
    //     return 1;
    // }

    // shuffleData(data.inputs, data.targets);

    const std::vector topology = {4096, 64, 3};
    MLP net(topology, 0.05);


    // std::cout << "Rozpoczynam trening..." << std::endl;
    // net.train(data.inputs, data.targets, 500);
    net.loadWeights("my_plants_v1.txt");
    cv::Mat testImg = cv::imread("../shefflera2.jpg", cv::IMREAD_GRAYSCALE);

    if (!testImg.empty()) {
        cv::Mat res;
        cv::resize(testImg, res, cv::Size(64,64));

        std::vector<float> input;
        input.reserve(4096);
        for (int r = 0; r < 64; ++r)
            for (int c = 0; c < 64; ++c)
                input.push_back(res.at<uchar>(r,c) / 255.0);

        net.feedForward(input);
        std::vector<float> out = net.getResults();

        std::cout << "\n--- TEST RECOGNITION ---" << std::endl;
        std::cout << "Gatunek 0 (Shifflera): " << out[0] * 100 << "%" << std::endl;
        std::cout << "Gatunek 1 (Pieniazek): " << out[1] * 100 << "%" << std::endl;
        std::cout << "Gatunek 2 (Hoya): " << out[2] * 100 << "%" << std::endl;

    }

    // net.saveWeights("my_plants_v1.txt");
    return 0;
}