#include "loss.h"
#include "activation.h"

namespace {

class QuadraticLoss : public Loss {
public:
    float loss(const Col& out, const Col& target) const override {
        const Col diff = out - target;
        return dot(diff, diff) * 0.5f;
    }

    Matrix delta(
            const Matrix& out,
            const Matrix& target,
            const Matrix& z) const override {
        return (out - target) % sigmoidDerivative(z);
    }
};

class CrossEntropyLoss : public Loss {
public:
    float loss(const Col& out, const Col& target) const override {
        Col x = -target % arma::log(out) - (1.0f - target) % arma::log(1 - out);
        x.replace(arma::datum::nan, 0.0f);
        return accu(x);
    }

    Matrix delta(
            const Matrix& out,
            const Matrix& target,
            const Matrix&) const override {
        return out - target;
    }
};

} // namespace

std::unique_ptr<Loss> quadraticLoss() {
    return std::make_unique<QuadraticLoss>();
}

std::unique_ptr<Loss> crossEntropyLoss() {
    return std::make_unique<CrossEntropyLoss>();
}
