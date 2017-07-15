#include <cmath>
#include <algorithm>
#include <iostream>
#include <random>

#include "types.h"
#include "mnist.h"

Matrix sigmoid(const Matrix& x) {
    return 1.0f / (1.0f + arma::exp(-x));
}

Matrix sigmoidDerivative(const Matrix& x) {
    const auto& s = sigmoid(x);
    return s % (1.0f - s);
}

size_t maxIndex(const Col& row) {
    return std::max_element(row.begin(), row.end()) - row.begin();
}

float loss(const Col& out, const Col& target) {
    const Col diff = out - target;
    return dot(diff, diff) * 0.5f;
}

Matrix lossDerivative(const Matrix& out, const Matrix& target) {
    return (out - target);
}

Col oneHot(size_t size, size_t i) {
    Col row(size);
    row.zeros();
    row(i) = 1.0f;
    return row;
}

struct PartialDerivatives {
    std::vector<Col> nablaBias;
    std::vector<Matrix> nablaWeights;
};

class NN;

struct EvaluationResult {
    float loss;
    float correctRatio;
};

EvaluationResult evaluate(
        const NN& nn,
        const std::vector<Sample>& samples);

class NN {
public:
    NN(std::vector<Col> bias, std::vector<Matrix> weights) :
        bias_(std::move(bias)), weights_(std::move(weights))
    {}

    Col feedforward(const Col& inputs) const {
        Col state = inputs;

        for (size_t i = 0; i != layers(); ++i) {
            state = sigmoid(weights_[i] * state + bias_[i]);
        }

        return state;
    }

    PartialDerivatives backprop(
            const Matrix& batch, const Matrix& batchTargets) {
        std::vector<Matrix> zs;
        std::vector<Matrix> activations = {batch};

        Matrix state = batch;
        for (size_t i = 0; i != layers(); ++i) {
            state = weights_[i] * state + repmat(bias_[i], 1, batch.n_cols);
            zs.push_back(state);
            state = sigmoid(state);
            activations.push_back(state);
        }
        activations.pop_back();

        Matrix delta = lossDerivative(state, batchTargets) %
            sigmoidDerivative(zs.back());
        std::vector<Col> nablaBias(bias_.size());
        std::vector<Matrix> nablaWeights(weights_.size());

        const float recip = 1.0f / batch.n_cols;
        nablaBias.back() = sum(delta, 1) * recip;
        nablaWeights.back() = delta * activations.back().t() * recip;

        for (size_t l = 1; l != layers(); ++l) {
            const size_t i = layers() - l - 1;
            const auto& z = zs[i];
            const auto sd = sigmoidDerivative(z);
            delta = weights_[i + 1].t() * delta % sd;
            nablaBias[i] = sum(delta, 1) * recip;
            nablaWeights[i] = delta * activations[i].t() * recip;
        }

        return {nablaBias, nablaWeights};
    }

    void sgd(
            std::vector<Sample> train,
            const std::vector<Sample>& test,
            size_t epochs,
            size_t miniBatchSize,
            float eta) {
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

                gradientStep(batchInput, batchTarget, eta);
            }
            const auto correctRatio = evaluate(*this, test).correctRatio;
            std::cout << "Epoch done: " << (epoch + 1) <<
                    ", correct: " << correctRatio <<
                    ", n = " << test.size() << '\n';
        }
    }

    size_t inputSize() const {
        return weights_.front().n_cols;
    }

    size_t outputSize() const {
        return weights_.back().n_rows;
    }

    size_t layers() const { return bias_.size(); }

public:
    void gradientStep(const Matrix& batchInput, const Matrix& batchTarget, float eta) {
        const auto partial = backprop(batchInput, batchTarget);

        for (size_t j = 0; j != layers(); ++j) {
            bias_[j] -= partial.nablaBias[j] * eta;
            weights_[j] -= partial.nablaWeights[j] * eta;
        }
    }

    std::vector<Col> bias_; // Bias for each of the layers
    std::vector<Matrix> weights_; // Connection weights
};

EvaluationResult evaluate(
        const NN& nn,
        const std::vector<Sample>& samples) {
    float loss = 0;
    size_t correct = 0;
    for (const auto& sample: samples) {
        const auto a = nn.feedforward(sample.x);
        const Col diff = a - oneHot(a.size(), sample.y);
        loss += dot(diff, diff);
        correct += (int)maxIndex(a) == sample.y;
    }
    return {
        loss / (2 * samples.size()),
        (float)correct / samples.size()};
}

class Builder {
public:
    explicit Builder(size_t inputSize) : lastLayerSize_(inputSize)
    {}

    void addFullyConnectedLayer(size_t size) {
        bias_.push_back(arma::randn<Col>(size));
        weights_.push_back(arma::randn<Matrix>(size, lastLayerSize_));
        lastLayerSize_ = size;
    }

    NN build() {
        return NN(bias_, weights_);
    }

    size_t lastLayerSize_;
    std::vector<Col> bias_;
    std::vector<Matrix> weights_;
};

int main()
{
    Builder builder(28 * 28);
    builder.addFullyConnectedLayer(30);
    builder.addFullyConnectedLayer(10);
    auto nn = builder.build();
    nn.sgd(
        mnist::readTrain(),
        mnist::readTest(),
        30,
        10,
        3.0f);
}
