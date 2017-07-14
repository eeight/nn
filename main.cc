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

Row oneHot(size_t size, size_t i) {
    Row row(size);
    row.zeros();
    row(i) = 1.0f;
    return row;
}

struct EvaluationResult {
    float loss;
    float correctRatio;
};

EvaluationResult evaluate(
        const NN& nn,
        const std::vector<Matrix>& inputs,
        const std::vector<int>& labels) {
    float loss = 0;
    size_t correct = 0;
    for (size_t i = 0; i != inputs.size(); ++i) {
        const auto a = nn.evaluate(vectorise(inputs[i], 1));
        loss += norm(a - oneHot(a.size(), labels[i]), 2);
        correct += (int)maxIndex(a) == labels[i];
    }
    return {
        loss / (2 * inputs.size()),
        (float)correct / inputs.size()};
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
    const auto result = evaluate(
            nn, mnist::readTest(), mnist::readTestLabels());

    std::cout << "Loss: " << result.loss <<
            ", correct: " << result.correctRatio << '\n';
}
