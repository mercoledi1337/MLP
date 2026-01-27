#ifndef MLP_LAYER_H
#define MLP_LAYER_H
#include <vector>
#include <cmath>

class Layer {
public:
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    std::vector<double> output;
    std::vector<double> lastInput;
    std::vector<double> deltas;

    Layer(int numNeurons, int numInputsPerNeuron);

    static double sigmoid(const double x) {
        return 1.0 / (1.0 + exp(-x));
    };

    static double sigmoidDerivative(const double x) {
        return x * (1.0 - x);
    }
    void forward(const std::vector<double>& inputs);
};

#endif //MLP_LAYER_H