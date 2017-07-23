#pragma once

#include "loss.h"
#include "train.h"
#include "types.h"
#include "tensor.h"
#include "program.h"

#include <vector>

using LossFunction = Tensor (*)(const Tensor&, const Tensor&);

class NN {
public:
    NN(
            Tensor input,
            Tensor output,
            std::vector<Tensor> bias,
            std::vector<Tensor> weights);

    Matrix predict(const Matrix& input) const;

    void fit(
            std::vector<Sample> train,
            const std::vector<Sample>& test,
            size_t epochs,
            float eta,
            LossFunction lossFunction,
            float lambda);

    size_t inputSize() const { return input_.shape().rows; }
    size_t outputSize() const { return output_.shape().rows; }
    size_t miniBatchSize() const { return input_.shape().cols; }

public:
    void gradientStep(
            const Matrix& input,
            const Matrix& target,
            Program& dLoss, float eta);

    Tensor input_;
    Tensor output_;
    std::vector<Tensor> bias_;
    std::vector<Tensor> weights_;
    std::vector<Tensor> params_;
    mutable Program eval_;
};
