#include <cmath>
#include <algorithm>
#include <iostream>
#include <random>

#include "activation.h"
#include "loss.h"
#include "mnist.h"
#include "types.h"
#include "nn.h"

class Builder {
public:
    explicit Builder(size_t inputSize, size_t miniBatchSize) :
        inputSize_(inputSize),
        lastLayerSize_(inputSize),
        miniBatchSize_(miniBatchSize),
    {}

    void addFullyConnectedLayer(size_t size) {
        bias_.push_back(newTensor(arma::randn<Col>(size)));
        weights_.push_back(newTensor(
                arma::randn<Matrix>(size, lastLayerSize_) /
                std::sqrt(static_cast<float>(lastLayerSize_))));
        lastLayerSize_ = size;
    }

    NN build() {
        auto input = newTensor(inputSize_, miniBatchSize);
        auto output = newTensor(lastLayerSize_, miniBatchSize_);

        Tensor state = input;
        for (size_t i = 0; i != bias_.size(); ++i) {
            state = weights_[i] * state + copyCols(bias_[i], miniBatchSize_);
            state = sigmoid(state);
        }

        return NN(input, output, miniBatchSize_, bias_, weights_);
    }

    size_t inputSize_;
    size_t lastLayerSize_;
    size_t miniBatchSize_;
    std::vector<Tensor> bias_;
    std::vector<Tensor> weights_;
};

int main()
{
    Builder builder(28 * 28);
    builder.addFullyConnectedLayer(100);
    builder.addFullyConnectedLayer(10);
    auto nn = builder.build();
    auto loss = crossEntropyLoss();
    nn.sgd(
        mnist::readTrain(),
        mnist::readTest(),
        30,
        10,
        0.5f,
        *loss,
        5.0);
}
