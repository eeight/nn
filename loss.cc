#include "loss.h"
#include "activation.h"

Tensor quadraticLoss(const Tensor& out, const Tensor& target) {
    return sumSquares(out - target) * 0.5;
}

Tensor crossEntropyLoss(const Tensor& out, const Tensor& target) {
    Tensor x = -target % log(out) - (1.0f - target) % log(1.0f - out);
    // FIXME
    // x.replace(arma::datum::nan, 0.0f);
    return sum(x);
}

} // namespace

