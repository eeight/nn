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

    TensorValue predict(const TensorValue& input) const;

    void fit(
            std::vector<Sample> train,
            const std::vector<Sample>& test,
            size_t epochs,
            float eta,
            LossFunction lossFunction,
            float lambda);

    Shape inputShape() const { return input_.shape().dropDim(); }
    Shape outputShape() const { return output_.shape().dropDim(); }
    size_t miniBatchSize() const { return input_.shape()(0); }

public:
    void gradientStep(
            const TensorValue& input,
            const TensorValue& target,
            Program& dLoss, float eta);

    Tensor input_;
    Tensor output_;
    std::vector<Tensor> bias_;
    std::vector<Tensor> weights_;
    std::vector<Tensor> params_;
    mutable Program eval_;
};
