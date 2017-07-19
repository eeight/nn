#pragma once

#include "types.h"
#include "expr.h"

#include <memory>

class Tensor {
public:
    Tensor() = delete;
    explicit Tensor(std::shared_ptr<Expr>);
    ~Tensor();

    Matrix eval(Ad *ad = nullptr) const;
    Shape shape() const;

    Tensor& operator +=(const Matrix& matrix);

    Tensor reshape(Shape shape) const;
    Tensor operator-() const;

    Tensor& operator=(const Matrix& matrix);
    void reset();

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
// Hadamard (Schur) division.
Tensor operator /(const Tensor& x, const Tensor& y);
Tensor operator +(const Tensor& x, const Tensor& y);
Tensor operator -(const Tensor& x, const Tensor& y);

// Computes x to the power of y.
// x must have shape (1, 1)
Tensor pow(const Tensor& x, float y);

// Element-wise exponent
Tensor exp(const Tensor& x);

// Element-wise logarithm
Tensor log(const Tensor& x);

// Compute partial derivatives of expr by each of the vars.
std::vector<Matrix> diff(const Tensor& expr, const std::vector<Tensor>& vars);

// Sums squares of all the elements in the tensor, returns scalar.
Tensor sumSquares(const Tensor& tensor);
// Sums all the elements in the tensor, returns scalar.
Tensor sum(const Tensor& tensor);
