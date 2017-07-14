#include <cmath>
#include <algorithm>
#include <iostream>

#include "types.h"
#include "mnist.h"

// Sigmoid activation function.
Row activate(const Row& x) {
    Row result = x;
    result.transform([](float x) { return 1.0f / (1.0f + std::exp(-x)); });
    return result;
}

size_t maxIndex(const Row& row) {
    return std::max_element(row.begin(), row.end()) - row.begin();
}

class NN {
public:
    NN(std::vector<Row> bias, std::vector<Matrix> weights) :
        bias_(std::move(bias)), weights_(std::move(weights))
    {}

    Row evaluate(const Row& inputs) const {
        Row state = inputs;

        for (size_t i = 0; i != weights_.size(); ++i) {
            state = activate(state * weights_[i] + bias_[i]);
        }

        return state;
    }

public:
    std::vector<Row> bias_; // Bias for each of the layers
    std::vector<Matrix> weights_; // Connection weights
};

float evaluate(
        const NN& nn,
        const std::vector<Matrix>& inputs,
        const std::vector<int>& labels) {
    size_t correct = 0;
    for (size_t i = 0; i != inputs.size(); ++i) {
        if ((int)maxIndex(nn.evaluate(vectorise(inputs[i], 1))) == labels[i]) {
            ++correct;
        }
    }
    return (float)correct/inputs.size();
}

class Builder {
public:
    explicit Builder(size_t inputSize) : lastLayerSize_(inputSize)
    {}

    void addFullyConnectedLayer(size_t size) {
        bias_.push_back(arma::randu<Row>(size));
        weights_.push_back(arma::randu<Matrix>(lastLayerSize_, size));
        lastLayerSize_ = size;
    }

    NN build() {
        return NN(bias_, weights_);
    }

    size_t lastLayerSize_;
    std::vector<Row> bias_;
    std::vector<Matrix> weights_;
};

int main()
{
    Builder builder(28 * 28);
    builder.addFullyConnectedLayer(100);
    auto nn = builder.build();

    std::cout << "Quality on test: " <<
        evaluate(nn, mnist::readTest(), mnist::readTestLabels()) << '\n';
}
