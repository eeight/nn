#include <cmath>
#include <algorithm>
#include <iostream>
#include <random>

#include "loss.h"
#include "mnist.h"
#include "types.h"
#include "nn.h"

#include <xmmintrin.h>

class Builder {
public:
    explicit Builder(size_t inputSize, size_t miniBatchSize) :
        input_(newTensor(inputSize, miniBatchSize)),
        output_(input_)
    {}

    void addFullyConnectedLayer(size_t size) {
        bias_.push_back(newTensor(arma::randn<Col>(size)));
        const size_t lastLayerSize = output_.shape().rows;
        weights_.push_back(newTensor(
                arma::randn<Matrix>(size, lastLayerSize) /
                std::sqrt(static_cast<float>(lastLayerSize))));
        output_ = sigmoid(weights_.back() * output_ + bias_.back());
    }

    NN build() {
        return NN(input_, output_, bias_, weights_);
    }

    Tensor input_;
    Tensor output_;
    std::vector<Tensor> bias_;
    std::vector<Tensor> weights_;
};

int main()
{
    _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);
    Builder builder(28 * 28, 10);
    builder.addFullyConnectedLayer(100);
    builder.addFullyConnectedLayer(10);
    auto nn = builder.build();
    nn.fit(
        mnist::readTrain(),
        mnist::readTest(),
        30,
        0.5f,
        &crossEntropyLoss,
        5.0);
}
