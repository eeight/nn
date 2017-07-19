#include "nn.h"

#include "activation.h"

namespace {

size_t maxIndex(const Col& row) {
    return std::max_element(row.begin(), row.end()) - row.begin();
}

float evaluate(
        const NN& nn, const std::vector<Sample>& samples) {
    size_t correct = 0;
    for (const auto& sample: samples) {
        const auto a = nn.feedforward(sample.x);
        correct += (int)maxIndex(a) == sample.y;
    }
    return (float)correct / samples.size();
}

} // namespace

Col NN::feedforward(const Col& inputs) const {
// FIXME figure out batching here
#if 0
    Col state = inputs;

    for (size_t i = 0; i != layers(); ++i) {
        state = sigmoid(weights_[i] * state + bias_[i]);
    }

    return state;
#endif
}

void NN::fit(
        std::vector<Sample> train,
        const std::vector<Sample>& test,
        size_t epochs,
        float eta,
        const Loss& loss,
        float lambda) {
    auto target = newTensor(outputSize(), miniBatchSize_);
    Tensor lossValue = loss(output_, target);
    std::mt19937 mt;
    for (size_t epoch = 0; epoch != epochs; ++epoch) {
        std::shuffle(train.begin(), train.end(), mt);
        for (size_t batch = 0; batch < train.size(); batch += miniBatchSize) {
            const size_t batchEnd = std::min(train.size(), batch + miniBatchSize);

            Matrix batchInput(inputSize(), batchEnd - batch);
            for (size_t i = batch; i != batchEnd; ++i) {
                batchInput.col(i - batch) = train[i].x;
            }

            Matrix batchTarget(outputSize(), batchEnd - batch);
            for (size_t i = batch; i != batchEnd; ++i) {
                batchTarget.col(i - batch) = oneHot(outputSize(), train[i].y);
            }

            gradientStep(batchInput, batchTarget, eta, loss, lambda, train.size());
        }
        const auto correctRatio = evaluate(*this, test, loss).correctRatio;
        std::cout << "Epoch done: " << (epoch + 1) <<
                ", correct: " << correctRatio <<
                ", n = " << test.size() << '\n';
    }
}

void NN::gradientStep(
        const Matrix& batchInput,
        const Matrix& batchTarget,
        float eta,
        const Loss& loss,
        float lambda,
        size_t n) {
    std::vector<Matrix> partial = diff(
            lossValue,
            weights_ and biases_,
            {
                {self.input, std::cref(batchInput)},
                {self.target, std::cref(batchTarget)}
            });

    for (size_t j = 0; j != layers(); ++j) {
        bias_[j] -= partial.nablaBias[j] * eta;
        weights_[j] = weights_[j] * (1.0f - eta * lambda / n) -
            partial.nablaWeights[j] * eta;
    }
}
