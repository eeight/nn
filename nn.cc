#include "nn.h"

#include "activation.h"

namespace {

size_t maxIndex(const Col& row) {
    return std::max_element(row.begin(), row.end()) - row.begin();
}

template <class F>
void forEachBatch(
        const std::vector<Sample>& samples,
        size_t batchSize,
        F f) {
    const size_t inputSize = samples.front().x.n_rows;
    const size_t outputSize = samples.front().y.n_rows;

    for (size_t i = 0; i + batchSize <= samples.size(); i += batchSize) {
        Matrix batchInput(inputSize, batchSize);
        Matrix batchTarget(outputSize, batchSize);
        for (size_t j = 0; j != batchSize; ++j) {
            batchInput.col(j) = samples[i + j].x;
            batchTarget.col(j) = samples[i + j].y;
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
        [&](const Matrix& batchInput, const Matrix& batchTarget) {
            const auto output = nn.predict(batchInput);
            for (size_t i = 0; i != batchTarget.n_cols; ++i) {
                correct +=
                    maxIndex(output.col(i)) == maxIndex(batchTarget.col(i));
                ++total;
            }
        });
    return (float)correct / total;
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
    weights_(std::move(weights))
{
    params_.insert(params_.end(), bias_.begin(), bias_.end());
    params_.insert(params_.end(), weights_.begin(), weights_.end());
}

Matrix NN::predict(const Matrix& input) {
    input_ = input;
    return output_.eval();
}

void NN::fit(
        std::vector<Sample> train,
        const std::vector<Sample>& test,
        size_t epochs,
        float eta,
        LossFunction lossFunction,
        float /*lambda*/) {
    auto target = newTensor(output_.shape());
#if 0
    Tensor regularizer = newTensor(arma::zeros<Matrix>(1, 1));
    for (const auto& w: weights_) {
        regularizer = regularizer + sumSquares(w);
    }
#endif
    Tensor loss = lossFunction(output_, target);
#if 0
        +
        lambda / (2 * train.size()) * regularizer;
#endif
    std::mt19937 mt;
    for (size_t epoch = 0; epoch != epochs; ++epoch) {
        std::shuffle(train.begin(), train.end(), mt);
        forEachBatch(
            train,
            miniBatchSize(),
            [&](const Matrix& batchInput, const Matrix& batchTarget) {
                input_ = batchInput;
                target = batchTarget;
                gradientStep(loss, eta);
            });
        const auto correctRatio = evaluate(*this, test);
        std::cout << "Epoch done: " << (epoch + 1) <<
                ", correct: " << correctRatio <<
                ", n = " << test.size() << '\n';
    }
}

void NN::gradientStep(const Tensor& loss, float eta) {
    auto partial = diff(loss, params_);

    for (size_t i = 0; i != params_.size(); ++i) {
        params_[i] += -eta * partial[i];
    }
    //std::cout << "loss: " << loss.eval()(0, 0) << '\n';
}
