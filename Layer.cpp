#include "Layer.h"
#include <random>
#include <omp.h>

Layer::Layer(const int numNeurons,const int numInputsPerNeuron) {
    output.resize(numNeurons);
    deltas.resize(numNeurons);
    biases.resize(numNeurons);
    weights.resize(numNeurons, std::vector<float>(numInputsPerNeuron));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution dis(-0.5, 0.5);

    for (int i = 0; i < numNeurons; i++) {
        biases[i] = dis(gen);
        for (int j = 0; j < numInputsPerNeuron; j++) {
            weights[i][j] = dis(gen);
        }
    }
}

void Layer::forward(const std::vector<float> &inputs) {
    this->lastInput = inputs;

    #pragma omp parallel for
    for (int i = 0; i < weights.size(); i++) {
        float sum = biases[i];
        for (int j = 0; j < inputs.size(); j++) {
            sum += inputs[j] * weights[i][j];
        }
        output[i] = sigmoid(sum);
    }
}