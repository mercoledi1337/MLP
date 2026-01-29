#ifndef MLP_MLP_H
#define MLP_MLP_H
#include <string>
#include <vector>
#include "Layer.h"

class MLP {
public:
    std::vector<Layer> layers;
    double learningRate;

    explicit MLP(const std::vector<int>& topology, double lr = 0.1);

    std::vector<float> feedForward(const std::vector<float>& input);

    void backpropagate(const std::vector<float>& target);

    void train(const std::vector<std::vector<float>>& inputs,
                      const std::vector<std::vector<float>>& targets,
                      int epochs = 1000);

    void saveWeights(std::string filename);

    [[nodiscard]] std::vector<float> getResults() const {
        return layers.back().output;
    };

    void loadWeights(std::string filename);
};
#endif //MLP_MLP_H