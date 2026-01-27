#include "MLP.h"
#include <iostream>
#include <ostream>

MLP::MLP(const std::vector<int> &topology,const double lr) : learningRate(lr) {
    for (size_t i = 1; i < topology.size(); ++i) {
        layers.push_back(Layer(topology[i], topology[i - 1]));
    }
}

std::vector<double> MLP::feedForward(const std::vector<double> &input) {
    std::vector<double> currentInput = input;
    for (auto& layer : layers) {
        layer.forward(currentInput);
        currentInput = layer.output;
    }
    return currentInput;
}


void MLP::backpropagate(const std::vector<double>& target) {
    Layer& outputLayer = layers.back();
    for (unsigned int i = 0; i < outputLayer.output.size(); ++i) {
        const double error = target[i] - outputLayer.output[i];
        outputLayer.deltas[i] = error * Layer::sigmoidDerivative(outputLayer.output[i]);
    }

    for (int i = layers.size() - 2; i >= 0; --i) {
        Layer& current = layers[i];
        Layer& next = layers[i + 1];

        for (int j = 0; j < current.output.size(); ++j) {
            double error = 0.0;
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

void MLP::train(const std::vector<std::vector<double>>& inputs,
                      const std::vector<std::vector<double>>& targets,
                      const int epochs) {
    for (int e = 0; e < epochs; ++e) {
        double totalError = 0.0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            std::vector<double> prediction = feedForward(inputs[i]);

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