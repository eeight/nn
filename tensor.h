#pragma once

#include "expr.h"

#include <functional>
#include <memory>

class Tensor {
public:
    Tensor() = delete;
    explicit Tensor(std::shared_ptr<Expr>);
    /* implicit */ Tensor(float x);
    ~Tensor();

    const Shape& shape() const;

    Tensor reshape(Shape shape) const;
    Tensor operator-() const;
    // Transposition
    Tensor t() const;
    // Reverse rows and columns order.
    Tensor r() const;

    Tensor vectorize() const;

    bool isConst1() const;

    const std::shared_ptr<Expr> unwrap() const { return expr_; }

private:
    std::shared_ptr<Expr> expr_;
};

void mutate(Tensor& t, const std::function<void (TensorValue&)>& mutator);

// Create placeholder variable. This variable can
// be given value only at an argument of compiled function.
Tensor newPlaceholder(const Shape& shape);

// Create variable with given value.
Tensor newTensor(TensorValue init);
Tensor newConstTensor(TensorValue init);

Tensor operator *(const Tensor& x, const Tensor& y);
// Hadamard (Schur) product.
Tensor operator %(const Tensor& x, const Tensor& y);
// Hadamard (Schur) division.
Tensor operator /(const Tensor& x, const Tensor& y);
Tensor operator +(const Tensor& x, const Tensor& y);
Tensor operator -(const Tensor& x, const Tensor& y);

// 2D convolution of a using kernel k result
// Inputs:
//  a: shape (mini batch size, n_rows, n_cols)
//  k: shape (n_features, n_kernel_rows, n_kernel_cols)
//  output: shape (mini batch size, n_features, n_result_rows, n_result_rows)
Tensor conv2d(const Tensor& a, const Tensor& kernel, bool sameSize);
Tensor conv2d(const Tensor& a, const Tensor& k, const Conv2D& conv);

Tensor maxPool2d(const Tensor& a, size_t rows, size_t cols);

// Computes x to the power of y.
// x must be a scalar
Tensor pow(const Tensor& x, float y);

// Element-wise exponent
Tensor exp(const Tensor& x);

// Element-wise logarithm
Tensor log(const Tensor& x);

// Element-wise sigmoid
Tensor sigmoid(const Tensor& x);

// Sums squares of all the elements in the tensor, returns scalar.
Tensor halfSumSquares(const Tensor& tensor);
// Sums all the elements in the tensor, returns scalar.
Tensor sum(const Tensor& tensor);
