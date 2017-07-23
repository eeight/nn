#pragma once

#include <mpark/variant.hpp>

#include "types.h"
#include "shape.h"

enum class BinaryOperator {
    Plus,
    Minus,
    Mul,
    HadamardMul,
    HadamardDiv
};

// Maybe the best ADT you can have in C++.
struct Const { Matrix value; };
struct Var { mpark::variant<Matrix, std::string> state;  };
struct Tile { size_t repeatRows; size_t repeatCols; Shape originalShape; };
struct Untile { size_t repeatRows; size_t repeatCols; Shape originalShape; };
struct BinaryOp { BinaryOperator op; };
struct Pow { float y; };
struct Exp {};
struct Log {};
struct Copy {};
struct Negate {};
// TODO prob needs mul - transpose fusion.
struct Transpose {};
struct Reshape { Shape shape; Shape originalShape; };
struct Sigmoid {};
struct SumSquares {};

using Op = mpark::variant<
    Const,
    Var,
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
    Sigmoid>;

struct Expr {
    template <class Op, class... Args>
    Expr(Shape shape, Op op, Args... args) :
        shape(shape), op(std::move(op)), args{std::move(args)...}
    {}

    Shape shape;
    Op op;
    std::vector<std::shared_ptr<Expr>> args;
};
