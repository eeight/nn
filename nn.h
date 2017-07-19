#pragma once

#include "loss.h"
#include "train.h"
#include "types.h"
#include "tensor.h"

#include <vector>

class NN {
public:
    NN(
            Tensor input,
            Tensor output,
            std::vector<Tensor> bias,
            std::vector<Tensor> weights) :
        input_(std::move(input)),
        output_(std::move(output)),
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
    std::vector<Tensor> bias_;
    std::vector<Tensor> weights_;
};
