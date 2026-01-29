#ifndef MLP_LAYER_H
#define MLP_LAYER_H
#include <vector>
#include <cmath>

class Layer {
public:
    std::vector<std::vector<float>> weights;
    std::vector<float> biases;
    std::vector<float> output;
    std::vector<float> lastInput;
    std::vector<float> deltas;

    Layer(int numNeurons, int numInputsPerNeuron);

    static float sigmoid(const float x) {
        return 1.0 / (1.0 + exp(-x));
    };

    static float sigmoidDerivative(const float x) {
        return x * (1.0 - x);
    }
    void forward(const std::vector<float>& inputs);
};
#endif //MLP_LAYER_H