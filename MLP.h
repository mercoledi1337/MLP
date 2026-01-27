#ifndef MLP_MLP_H
#define MLP_MLP_H
#include <vector>
#include "Layer.h"

class MLP {
public:
    std::vector<Layer> layers;
    double learningRate;

    explicit MLP(const std::vector<int>& topology, double lr = 0.1);

    std::vector<double> feedForward(const std::vector<double>& input);

    void backpropagate(const std::vector<double>& target);

    void train(const std::vector<std::vector<double>>& inout,
                      const std::vector<std::vector<double>>& target,
                      int epochs = 1000);
};

#endif //MLP_MLP_H