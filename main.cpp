#include "Layer.h"
#include "MLP.h"
#include <iostream>

int main() {

    std::vector topology = {2, 4, 1};
    MLP net(topology, 0.5);

    std::vector<std::vector<double>> inputs = {
        {0.0, 0.0},
        {0.1, 1.0},
        {1.0, 0.0},
        {1.1, 1.1}
    };

    std::vector<std::vector<double>> targets = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    std::cout << "Rozpoczynam trening..." << std::endl;
    net.train(inputs, targets, 10000);

    std::cout << "\nWyniki po treningu:" << std::endl;
    for (const auto& in : inputs) {
        std::vector<double> out = net.feedForward(in);
        std::cout << in[0] << " XOR " << in[1] << " = " << out[0] << std::endl;
    }
    return 0;
}