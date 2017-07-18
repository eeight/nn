#pragma once

#include "types.h"

#include <memory>

namespace t {

class Tensor;
class Expr;

class Tensor {
public:
    Tensor() = delete;
    explicit Tensor(std::shared_ptr<Expr>);
    ~Tensor();

    Matrix eval() const;
    size_t rows() const;
    size_t cols() const;

    Tensor& operator +=(const Matrix& matrix);

    friend const std::shared_ptr<Expr>& unwrap(const Tensor&);

private:
    std::shared_ptr<Expr> expr_;
};

Tensor newTensor(size_t rows, size_t cols);
Tensor newTensor(Matrix init);
Tensor newConstTensor(Matrix init);

Tensor operator *(const Tensor& x, const Tensor& y);
// Hadamard (Schur) product.
Tensor operator %(const Tensor& x, const Tensor& y);
Tensor operator +(const Tensor& x, const Tensor& y);
Tensor operator -(const Tensor& x, const Tensor& y);

// Computes x to the power of y.
// x must have shape (1, 1)
Tensor pow(const Tensor&x, float y);

// Compute partial derivatives of expr by each of the vars.
std::vector<Matrix> diff(const Tensor& expr, const std::vector<Tensor>& vars);

} // namspace t
