#include "neural.h"
#include <iostream>

int main() {
    // XOR problem
    NeuralNetwork nn({2, 2, 1});
    std::vector<std::vector<double>> inputs = {
        {0,0}, {0,1}, {1,0}, {1,1}
    };
    std::vector<std::vector<double>> targets = {
        {0}, {1}, {1}, {0}
    };
    nn.train(inputs, targets, 0.5, 5000);

    std::cout << "Testing XOR:\n";
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto out = nn.predict(inputs[i]);
        std::cout << inputs[i][0] << " xor " << inputs[i][1] << " = " << out[0] << "\n";
    }
    return 0;
}
