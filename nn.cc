#include "nn.h"

#include "ad.h"

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

template <class T>
std::vector<T> concat(const std::vector<T>& x, const std::vector<T>& y) {
    std::vector<T> result = x;
    result.insert(result.end(), y.begin(), y.end());
    return result;
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
    weights_(std::move(weights)),
    params_(concat(bias_, weights_)),
    eval_(compile({output}, {"x"}))
{
}

Matrix NN::predict(const Matrix& input) const{
    return eval_({&input}).front();
}

void NN::fit(
        std::vector<Sample> train,
        const std::vector<Sample>& test,
        size_t epochs,
        float eta,
        LossFunction lossFunction,
        float lambda) {
    auto target = newTensor("y", output_.shape());
    Tensor regularizer = newTensor(arma::zeros<Matrix>(1, 1));
    for (const auto& w: weights_) {
        regularizer = regularizer + sumSquares(w);
    }
    Tensor loss =
        lossFunction(output_, target) / miniBatchSize() +
        lambda / (2 * train.size()) * regularizer;
    auto dLoss = compile(diff(loss, params_), {"x", "y"});
    std::mt19937 mt;
    for (size_t epoch = 0; epoch != epochs; ++epoch) {
        std::shuffle(train.begin(), train.end(), mt);
        forEachBatch(
            train,
            miniBatchSize(),
            [&](const Matrix& batchInput, const Matrix& batchTarget) {
                gradientStep(batchInput, batchTarget, dLoss, eta);
            });
        const auto correctRatio = evaluate(*this, test);
        std::cout << "Epoch done: " << (epoch + 1) <<
                ", correct: " << correctRatio <<
                ", n = " << test.size() << '\n';
    }
}

void NN::gradientStep(
        const Matrix& input,
        const Matrix& target,
        Program& dLoss,
        float eta) {
    auto partial = dLoss({&input, &target});

    for (size_t i = 0; i != params_.size(); ++i) {
        params_[i] += -eta * partial[i];
    }
}
