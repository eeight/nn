#pragma once

#include <mpark/variant.hpp>

#include "types.h"
#include "shape.h"
#include "tensor_value.h"

enum class BinaryOperator {
    Plus,
    Mul,
    HadamardMul,
    HadamardDiv
};

// Maybe the best ADT you can have in C++.
struct Const { TensorValue value; };
struct Var { TensorValue value;  };
struct Placeholder {};
struct Tile { Shape multiplier; };
struct Untile { Shape multiplier; };
struct BinaryOp { BinaryOperator op; };
struct Pow { float y; };
struct Exp {};
struct Log {};
struct Copy {};
struct Negate {};
struct Transpose {};
struct Reshape {};
struct Sigmoid {};
struct HalfSumSquares {};
struct Reverse {};
struct MaxPool { Shape pool; };
struct MaxPoolDiff { Shape pool; };

using Op = mpark::variant<
    Const,
    Var,
    Placeholder,
    Tile,
    Untile,
    BinaryOp,
    Pow,
    Exp,
    Log,
    Copy,
    Negate,
    Transpose,
    Reshape,
    Sigmoid,
    HalfSumSquares,
    Conv2D,
    Reverse,
    MaxPool,
    MaxPoolDiff>;

struct Expr {
    template <class Op, class... Args>
    Expr(Shape shape, Op op, Args... args) :
        shape(shape), op(std::move(op)), args{std::move(args)...}
    {}

    Shape shape;
    Op op;
    std::vector<std::shared_ptr<Expr>> args;
};
