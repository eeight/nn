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

struct Const { Matrix value; };
struct Var { mpark::variant<Matrix, std::string> state;  };
struct Tile { size_t repeatCols; size_t repeatRows; Shape originalShape; };
struct BinaryOp { BinaryOperator op; };
struct Pow { float y; };
struct Exp {};
struct Log {};
struct Copy {};
struct Negate {};
struct Reshape { Shape shape; Shape originalShape; };

using Op = mpark::variant<
        Const, Var, Tile, BinaryOp, Pow, Exp, Log, Copy, Negate, Reshape>;

struct Expr {
    template <class Op, class... Args>
    Expr(Shape shape, Op op, Args... args) :
        shape(shape), op(std::move(op)), args{std::move(args)...}
    {}

    Shape shape;
    Op op;
    std::vector<std::shared_ptr<Expr>> args;
};
