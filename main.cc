#include <cmath>
#include <algorithm>
#include <iostream>
#include <random>

#include "types.h"
#include "mnist.h"

Col sigmoid(const Col& x) {
    return 1.0f / (1.0f + arma::exp(-x));
}

Col sigmoidDerivative(const Col& x) {
    const Col s = sigmoid(x);
    return s % (1.0f - s);
}

size_t maxIndex(const Col& row) {
    return std::max_element(row.begin(), row.end()) - row.begin();
}

float loss(const Col& out, const Col& target) {
    const Col diff = out - target;
    return dot(diff, diff);
}

Col lossDerivative(const Col& out, const Col& target) {
    return 2.0f * (out - target);
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

        for (size_t i = 0; i != weights_.size(); ++i) {
            state = sigmoid(weights_[i] * state + bias_[i]);
        }

        return state;
    }

    PartialDerivatives backprop(const Col& input, const Col& target) {
        std::vector<Col> zs;
        std::vector<Col> activations = {input};

        Col state = input;
        for (size_t i = 0; i != weights_.size(); ++i) {
            state = weights_[i] * state + bias_[i];
            zs.push_back(state);
            state = sigmoid(state);
            activations.push_back(state);
        }
        activations.pop_back();

        Col delta = lossDerivative(state, target) %
            sigmoidDerivative(zs.back());
        std::vector<Col> nablaBias(bias_.size());
        std::vector<Matrix> nablaWeights(weights_.size());

        nablaBias.back() = delta;
        nablaWeights.back() = delta * activations.back().t();

        for (size_t l = 1; l != bias_.size(); ++l) {
            const size_t i = bias_.size() - l - 1;
            const auto& z = zs[i];
            const auto sd = sigmoidDerivative(z);
            delta = weights_[i + 1].t() * delta % sd;
            nablaBias[i] = delta;
            nablaWeights[i] = delta * activations[i].t();
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
            for (size_t i = 0; i < train.size(); i += miniBatchSize) {
                const size_t iEnd = std::min(train.size(), i + miniBatchSize);
                gradientStep(train.begin() + i, train.begin() + iEnd, eta);
            }
            const auto correctRatio = evaluate(*this, test).correctRatio;
            std::cout << "Epoch done: " << (epoch + 1) <<
                    ", correct: " << correctRatio <<
                    ", n = " << test.size() << '\n';
        }
    }

    size_t outputSize() const {
        return weights_.back().n_rows;
    }

public:
    template <class Iterator>
    void gradientStep(Iterator begin, Iterator end, float eta) {
        std::vector<Col> nablaBias;
        nablaBias.reserve(bias_.size());
        for (const auto& b: bias_) {
            nablaBias.push_back(arma::zeros<Col>(size(b)));
        }

        std::vector<Matrix> nablaWeights;
        nablaWeights.reserve(weights_.size());
        for (const auto& w: weights_) {
            nablaWeights.push_back(arma::zeros<Matrix>(size(w)));
        }

        for (auto i = begin; i != end; ++i) {
            const auto& sample = *i;
            const auto partial = backprop(
                    sample.x, oneHot(outputSize(), sample.y));
            for (size_t j = 0; j != bias_.size(); ++j) {
                nablaBias[j] += partial.nablaBias[j];
                nablaWeights[j] += partial.nablaWeights[j];
            }
        }

        const float factor = eta / (end - begin);
        for (size_t j = 0; j != bias_.size(); ++j) {
            bias_[j] -= nablaBias[j] * factor;
            weights_[j] -= nablaWeights[j] * factor;
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
        6.0f);
}
