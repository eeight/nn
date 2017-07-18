#pragma once

#include "loss.h"
#include "train.h"
#include "types.h"

#include <vector>

struct PartialDerivatives {
    std::vector<Col> nablaBias;
    std::vector<Matrix> nablaWeights;
};

class NN {
public:
    NN(
            Tensor input,
            Tensor output,
            size_t miniBatchSize,
            std::vector<Tensor> bias,
            std::vector<Tensor> weights) :
        input_(std::move(input)),
        output_(std::move(output)),
        miniBatchSize_(miniBatchSize),
        bias_(std::move(bias)),
        weights_(std::move(weights))
    {}

    Col feedforward(const Col& input) const;

    void fit(
            std::vector<Sample> train,
            const std::vector<Sample>& test,
            size_t epochs,
            float eta,
            const Loss& loss,
            float lambda);

    size_t inputSize() const {
        return input_.rows();
    }

    size_t outputSize() const {
        return ouput_.rows();
    }

public:
    void gradientStep(
            const Matrix& batchInput,
            const Matrix& batchTarget,
            float eta,
            const Loss& loss,
            float lambda,
            size_t n);

    Tensor input_;
    Tensor output_;
    size_t miniBatchSize_;
    std::vector<Tensor> bias_;
    std::vector<Tensor> weights_;
};
