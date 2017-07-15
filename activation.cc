#include "activation.h"

Matrix sigmoid(const Matrix& x) {
    return 1.0f / (1.0f + arma::exp(-x));
}

Matrix sigmoidDerivative(const Matrix& x) {
    const auto& s = sigmoid(x);
    return s % (1.0f - s);
}

