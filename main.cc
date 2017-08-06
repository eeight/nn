#include <cmath>
#include <algorithm>
#include <iostream>
#include <random>

#include "loss.h"
#include "mnist.h"
#include "nn.h"

#include <xmmintrin.h>

class Builder {
public:
    explicit Builder(const Shape& inputShape, size_t miniBatchSize) :
        input_(newPlaceholder(inputShape.addDim(miniBatchSize))),
        output_(input_)
    {}

    void addFullyConnectedLayer(size_t size) {
        bias_.push_back(newTensor(TensorValue::randn({1, size})));
        const size_t lastLayerSize = output_.shape().dropDim().size();
        const size_t miniBatchSize = input_.shape()(0);
        weights_.push_back(newTensor(
                    TensorValue::randn(
                        {lastLayerSize, size},
                        1.0f / std::sqrt(static_cast<float>(lastLayerSize)))));
        output_ = sigmoid(
                output_.reshape({miniBatchSize, lastLayerSize}) * weights_.back() +
                bias_.back());
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
    Builder builder({28 * 28}, 10);
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
