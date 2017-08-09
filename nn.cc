#include "nn.h"

#include "ad.h"

#include <iostream>
#include <random>

namespace {

size_t maxIndex(const float* x, size_t size) {
    return std::max_element(x, x + size) - x;
}

template <class F>
void forEachBatch(
        const std::vector<Sample>& samples,
        size_t batchSize,
        F f) {
    const size_t inputSize = samples.front().x.shape()(0);
    const size_t outputSize = samples.front().y.shape()(0);

    for (size_t i = 0; i + batchSize <= samples.size(); i += batchSize) {
        auto batchInput = TensorValue::zeros({batchSize, inputSize});
        auto batchTarget = TensorValue::zeros({batchSize, outputSize});
        for (size_t j = 0; j != batchSize; ++j) {
            std::copy(
                    samples[i + j].x.data(),
                    samples[i + j].x.dataEnd(),
                    &batchInput(j, 0));
            std::copy(
                    samples[i + j].y.data(),
                    samples[i + j].y.dataEnd(),
                    &batchTarget(j, 0));
        }

        f(batchInput, batchTarget);
    }
}

float evaluate(
        NN& nn, const std::vector<Sample>& samples) {
    size_t correct = 0;
    size_t total = 0;
    forEachBatch(
        samples,
        nn.miniBatchSize(),
        [&](const TensorValue& batchInput, const TensorValue& batchTarget) {
            const auto output = nn.predict(batchInput);
            const size_t rows = batchTarget.shape()(0);
            const size_t cols = batchTarget.shape()(1);
            for (size_t i = 0; i != rows; ++i) {
                correct +=
                        maxIndex(&output(i, 0), cols) ==
                        maxIndex(&batchTarget(i, 0), cols);
                ++total;
            }
        });
    return (float)correct / total;
}

template <class T>
std::vector<T> concat(const std::vector<T>& x, const std::vector<T>& y) {
    std::vector<T> result = x;
    result.insert(result.end(), y.begin(), y.end());
    return result;
}

} // namespace

NN::NN(
        Tensor input,
        Tensor output,
        std::vector<Tensor> bias,
        std::vector<Tensor> weights) :
    input_(std::move(input)),
    output_(std::move(output)),
    bias_(std::move(bias)),
    weights_(std::move(weights)),
    params_(concat(bias_, weights_)),
    eval_(compile({output}, {input_}))
{
    std::cout << "Eval: " << eval_;
}

TensorValue NN::predict(const TensorValue& input) const{
    return eval_({&input}).front();
}

void NN::fit(
        std::vector<Sample> train,
        const std::vector<Sample>& test,
        size_t epochs,
        float eta,
        LossFunction lossFunction,
        float lambda) {
    auto target = newPlaceholder(output_.shape());
    Tensor regularizer = newTensor(TensorValue{0.0f});
    for (const auto& w: weights_) {
        regularizer = regularizer + halfSumSquares(w);
    }
    Tensor loss =
        lossFunction(output_, target) / miniBatchSize() +
        lambda / train.size() * regularizer;
    auto dLoss = compile(diff(loss, params_), {input_, target});
    std::cout << "Loss: " << compile({loss}, {input_, target});
    std::cout << "dLoss: " << dLoss;
    std::default_random_engine generator;
    for (size_t epoch = 0; epoch != epochs; ++epoch) {
        std::shuffle(train.begin(), train.end(), generator);
        forEachBatch(
            train,
            miniBatchSize(),
            [&](const TensorValue& batchInput, const TensorValue& batchTarget) {
                gradientStep(batchInput, batchTarget, dLoss, eta);
            });
        const auto correctRatio = evaluate(*this, test);
        std::cout << "Epoch done: " << (epoch + 1) <<
                ", correct: " << correctRatio <<
                ", n = " << test.size() << '\n';
    }
}

void NN::gradientStep(
        const TensorValue& input,
        const TensorValue& target,
        Program& dLoss,
        float eta) {
    const auto& partial = dLoss({&input, &target});

    for (size_t i = 0; i != params_.size(); ++i) {
        mutate(params_[i], [&](TensorValue& param) {
            addMultiply(partial[i], -eta, &param);
        });
    }
}
