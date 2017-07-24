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
    Tensor t() const;

    bool isConst1() const;

    const std::shared_ptr<Expr> unwrap() const { return expr_; }

private:
    std::shared_ptr<Expr> expr_;
};

void mutate(Tensor& t, const std::function<void (Matrix&)>& mutator);

// Create placeholder variable with given name. This variable can
// be given value only at an argument of compiled function.
Tensor newTensor(std::string name, size_t rows, size_t cols);
Tensor newTensor(std::string name, Shape shape);

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
