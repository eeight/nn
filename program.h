#pragma once

#include "tensor.h"

#include <mpark/variant.hpp>

#include <deque>
#include <vector>

namespace detail {

struct ArgRef {
    size_t index;
    Shape shape;
};

using ReadRef = mpark::variant<ArgRef, ConstTensorRef>;

struct NegateTranspose {
    bool negate = false;
    bool transpose = false;
};

struct FusedBinaryOp {
    BinaryOperator op;
    NegateTranspose xMod;
    NegateTranspose yMod;
};

using VmOp = mpark::variant<
    Tile,
    Untile,
    FusedBinaryOp,
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

struct Statement {
    VmOp op;
    std::vector<ReadRef> args;
    TensorRef result;
};

} // namespace detail

class Program {
public:
    friend class ResolutionVisitor;
    const std::vector<TensorValue>& operator()(
            const std::vector<const TensorValue*>& args = {});

    Program(
            std::vector<detail::Statement> program,
            std::deque<TensorValue> tmp,
            std::vector<TensorValue> result,
            std::vector<std::shared_ptr<Expr>> retainer) :
        program_(std::move(program)),
        tmp_(std::move(tmp)),
        result_(std::move(result)),
        retainer_(std::move(retainer))
    {}

    friend std::ostream& operator <<(std::ostream&, const Program&);

private:
    std::vector<detail::Statement> program_;
    // TODO(eeight) reuse tmp space
    std::deque<TensorValue> tmp_;
    std::vector<TensorValue> result_;
    std::vector<std::shared_ptr<Expr>> retainer_;
};

Program compile(
            const std::vector<Tensor>& targets,
            const std::vector<Tensor>& args);
