#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

#include "loss.h"
#include "mnist.h"
#include "nn.h"

#include <xmmintrin.h>

class Builder {
public:
    explicit Builder(const Shape& inputShape, size_t miniBatchSize) :
        input_(newPlaceholder(inputShape.addFirstDim(miniBatchSize))),
        output_(input_)
    {}

    void addConvoLayer(size_t kernelSize, size_t features, size_t maxPoolSize) {
        auto kernel = newTensor(TensorValue::randn(
                    {features, kernelSize, kernelSize}));
        output_ = conv2d(output_, kernel, /* sameSize = */false);
        // A random scalar.
        auto bias = newTensor(TensorValue::randn(Shape{}));
        output_ = sigmoid(output_ + bias);
        output_ = maxPool2d(output_, maxPoolSize, maxPoolSize);
        params_.push_back(kernel);
        params_.push_back(bias);
    }

    void addFullyConnectedLayer(size_t size) {
        const size_t lastLayerSize = output_.shape().dropFirstDim().size();
        const size_t miniBatchSize = input_.shape()(0);
        auto bias = newTensor(TensorValue::randn({1, size}));
        auto weights = newTensor(
            TensorValue::randn(
                {lastLayerSize, size},
                1.0f / std::sqrt(static_cast<float>(lastLayerSize))));
        output_ = sigmoid(
                output_.reshape({miniBatchSize, lastLayerSize}) * weights +
                bias);
        params_.push_back(bias);
        params_.push_back(weights);
    }

    NN build() {
        return NN(input_, output_, params_);
    }

    Tensor input_;
    Tensor output_;
    std::vector<Tensor> params_;
};

class Listener : public FittingListener {
    static constexpr int length = 50;

public:
    explicit Listener(size_t epochs) : epochs_(epochs)
    {
        if (epochs_) {
            initEpochDisplay(0);
        }
    }

    void onEpochDone(size_t epoch, float accuracy) override {
        std::cout << "] done; accuracy = " << accuracy << '\n';
        if (epoch + 1 < epochs_) {
            initEpochDisplay(epoch + 1);
        }
    }

    void onBatchDone(float progress) override {
        const int displayedProgress = length * progress;
        if (displayedProgress == displayedProgress_) {
            return;
        }
        std::cout << std::string(displayedProgress - displayedProgress_, '=');
        std::cout << std::flush;
        displayedProgress_ = displayedProgress;
    }

private:
    void initEpochDisplay(size_t epoch) {
        displayedProgress_ = 0;
        auto printHeading = [&] {
            std::cout << "Epoch " << std::setw(4) << epoch  << ": [";
        };
        printHeading();
        std::cout << std::string(length, ' ');
        std::cout << "]\r";
        printHeading();
        std::cout << std::flush;
    }

    size_t epochs_;
    float displayedProgress_;
};

int main()
{
    _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);
    Builder builder({28, 28}, 10);
    builder.addConvoLayer(5, 3, 2);
    builder.addFullyConnectedLayer(100);
    builder.addFullyConnectedLayer(10);
    auto nn = builder.build();
    const size_t epochs = 30;
    Listener listener(epochs);
    nn.fit(
        mnist::readTrain(),
        mnist::readTest(),
        epochs,
        0.5f,
        &crossEntropyLoss,
        5.0,
        &listener);
}
