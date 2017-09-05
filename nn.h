#pragma once

#include "loss.h"
#include "train.h"
#include "tensor.h"
#include "program.h"

#include <vector>

using LossFunction = Tensor (*)(const Tensor&, const Tensor&);

class FittingListener {
public:
    virtual ~FittingListener() = default;

    virtual void onEpochDone(size_t epoch, float accuracy) = 0;
    virtual void onBatchDone(float progress) = 0;
};

class NN {
public:
    NN(Tensor input, Tensor output, std::vector<Tensor> paras);

    TensorValue predict(const TensorValue& input) const;

    void fit(
            std::vector<Sample> train,
            const std::vector<Sample>& test,
            size_t epochs,
            float eta,
            LossFunction lossFunction,
            float lambda,
            FittingListener* listener);

    Shape inputShape() const { return input_.shape().dropFirstDim(); }
    Shape outputShape() const { return output_.shape().dropFirstDim(); }
    size_t miniBatchSize() const { return input_.shape()(0); }

public:
    void gradientStep(
            const TensorValue& input,
            const TensorValue& target,
            Program& dLoss, float eta);

    Tensor input_;
    Tensor output_;
    std::vector<Tensor> params_;
    mutable Program eval_;
};
