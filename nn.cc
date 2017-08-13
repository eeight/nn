#include "nn.h"

#include "ad.h"

#include <iostream>
#include <random>

namespace {

size_t maxIndex(const float* x, size_t size) {
    return std::max_element(x, x + size) - x;
}

template <class F>
void forEachBatch(const std::vector<Sample>& samples, size_t batchSize, F f) {
    auto batchInput = TensorValue::zeros(
            samples.front().x.shape().addFirstDim(batchSize));
    auto batchTarget = TensorValue::zeros(
            samples.front().y.shape().addFirstDim(batchSize));
    const size_t sampleInputSize = samples.front().x.shape().size();
    const size_t sampleOutputSize = samples.front().y.shape().size();

    for (size_t i = 0; i + batchSize <= samples.size(); i += batchSize) {
        for (size_t j = 0; j != batchSize; ++j) {
            std::copy(
                    samples[i + j].x.data(),
                    samples[i + j].x.dataEnd(),
                    batchInput.data() + j * sampleInputSize);
            std::copy(
                    samples[i + j].y.data(),
                    samples[i + j].y.dataEnd(),
                    batchTarget.data() + j * sampleOutputSize);
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
                correct += maxIndex(&output(i, 0), cols) ==
                        maxIndex(&batchTarget(i, 0), cols);
                ++total;
            }
        });
    return (float)correct / total;
}

} // namespace

NN::NN(Tensor input, Tensor output, std::vector<Tensor> params) :
    input_(std::move(input)),
    output_(std::move(output)),
    params_(std::move(params)),
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
    for (const auto& w: params_) {
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
