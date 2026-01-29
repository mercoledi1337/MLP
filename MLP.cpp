#include "MLP.h"
#include <fstream>
#include <iostream>
#include <ostream>


MLP::MLP(const std::vector<int> &topology, const double lr) : learningRate(lr) {
    for (size_t i = 1; i < topology.size(); ++i) {
        layers.push_back(Layer(topology[i], topology[i - 1]));
    }
}

std::vector<float> MLP::feedForward(const std::vector<float> &input) {
    std::vector<float> currentInput = input;
    for (auto& layer : layers) {
        layer.forward(currentInput);
        currentInput = layer.output;
    }
    return currentInput;
}

void MLP::backpropagate(const std::vector<float>& target) {
    Layer& outputLayer = layers.back();
    for (unsigned int i = 0; i < outputLayer.output.size(); ++i) {
        const float error = target[i] - outputLayer.output[i];
        outputLayer.deltas[i] = error * Layer::sigmoidDerivative(outputLayer.output[i]);
    }

    for (int i = layers.size() - 2; i >= 0; --i) {
        Layer& current = layers[i];
        Layer& next = layers[i + 1];

        for (int j = 0; j < current.output.size(); ++j) {
            float error = 0.0;
            for (int k = 0; k < next.output.size(); ++k) {
                error += next.deltas[k] * next.weights[k][j];
            }
            current.deltas[j] = error * Layer::sigmoidDerivative(current.output[j]);
        }
    }
    for (auto& layer : layers) {
        for (int i = 0; i < layer.weights.size(); ++i) {
                for (int j = 0; j < layer.weights[i].size(); ++j) {
                    layer.weights[i][j] += learningRate * layer.deltas[i] * layer.lastInput[j];
                }
            layer.biases[i] += learningRate * layer.deltas[i];
        }
    }
}

void MLP::train(const std::vector<std::vector<float>>& inputs,
                      const std::vector<std::vector<float>>& targets,
                      const int epochs) {
    for (int e = 0; e < epochs; ++e) {
        float totalError = 0.0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            std::vector<float> prediction = feedForward(inputs[i]);

            for (size_t j = 0; j < targets[i].size(); ++j) {
                totalError += std::pow(targets[i][j] - prediction[j], 2);
            }

            backpropagate(targets[i]);
        }
        if (e % 100 == 0) {
            std::cout << "Epoch: " << e << "/ mean erros" << totalError / inputs.size() << std::endl;
        }
    }
}

void MLP::saveWeights(std::string filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cout << "Can't open file" << std::endl;
        return;
    }

    for (auto &layer : layers) {
        for (const auto &neuronWeights : layer.weights) {
            for (float w : neuronWeights) {
                file << w << " ";
            }
            file << "\n";
        }
        for (float b : layer.biases) {
            file << b << " ";
        }
        file << "\n";
    }

    file.close();
    std::cout << "Saved weights to " << filename << std::endl;
};

void MLP::loadWeights(std::string filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Blad: Nie mozna otworzyc pliku " << filename << " do odczytu!" << std::endl;
        return;
    }

    for (auto &layer : layers) {

        for (auto &neuronWeights : layer.weights) {
            for (float &w : neuronWeights) {
                if (!(file >> w)) break;
            }
        }

        for (float &b : layer.biases) {
            if (!(file >> b)) break;
        }
    }

    file.close();
    std::cout << "--- Wagi wczytane pomyslnie z pliku: " << filename << " ---" << std::endl;
}