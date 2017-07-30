#pragma once

#include "types.h"
#include "expr.h"

#include <functional>
#include <memory>

class Tensor {
public:
    Tensor() = delete;
    explicit Tensor(std::shared_ptr<Expr>);
    /* implicit */ Tensor(float x);
    ~Tensor();

    Shape shape() const;

    Tensor reshape(Shape shape) const;
    Tensor operator-() const;
    // Transposition
    Tensor t() const;
    // Reverse rows and columns order.
    Tensor r() const;

    bool isConst1() const;

    const std::shared_ptr<Expr> unwrap() const { return expr_; }

private:
    std::shared_ptr<Expr> expr_;
};

void mutate(Tensor& t, const std::function<void (Matrix&)>& mutator);

// Create placeholder variable. This variable can
// be given value only at an argument of compiled function.
Tensor newTensor(size_t rows, size_t cols);
Tensor newTensor(Shape shape);

// Create variable with given value. Its value can be mutated
// by operators = and +=.
Tensor newTensor(Matrix init);
Tensor newConstTensor(Matrix init);

Tensor operator *(const Tensor& x, const Tensor& y);
// Hadamard (Schur) product.
Tensor operator %(const Tensor& x, const Tensor& y);
// Hadamard (Schur) division.
Tensor operator /(const Tensor& x, const Tensor& y);
Tensor operator +(const Tensor& x, const Tensor& y);
Tensor operator -(const Tensor& x, const Tensor& y);

// 2D convolution of a using kernel k result.
Tensor conv2d(const Tensor& a, const Tensor& kernel, bool sameSize);
// Same as above but with explicit padding.
Tensor conv2d(
        const Tensor& a,
        const Tensor& k,
        const Conv2D& conv);

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
