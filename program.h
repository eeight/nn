#pragma once

#include "tensor.h"

#include <vector>
#include <mpark/variant.hpp>

namespace detail {

struct ArgRef { size_t index; };
struct ResultRef { size_t index; };
struct TmpRef { size_t index; };
struct VarRef { const Matrix* matrix; };

using ReadRef = mpark::variant<ArgRef, ResultRef, TmpRef, VarRef>;
using WriteRef = mpark::variant<ResultRef, TmpRef>;

struct Statement {
    Op op;
    std::vector<ReadRef> args;
    WriteRef result;
};

} // namespace detail

class Program {
public:
    friend class ResolutionVisitor;
    const std::vector<Matrix>& operator()(
            const std::vector<const Matrix*>& args = {});

    Program(
            std::vector<detail::Statement> program,
            std::vector<Matrix> tmp,
            std::vector<Matrix> result,
            std::vector<std::shared_ptr<Expr>> retainer) :
        program_(std::move(program)),
        tmp_(std::move(tmp)),
        result_(std::move(result)),
        retainer_(std::move(retainer))
    {}

private:
    void execute(
            const detail::Statement& stmt,
            const std::vector<const Matrix *>& args);

    std::vector<detail::Statement> program_;
    // TODO(eeight) reuse tmp space
    std::vector<Matrix> tmp_;
    std::vector<Matrix> result_;
    std::vector<std::shared_ptr<Expr>> retainer_;
};

Program compile(
            const std::vector<Tensor>& targets,
            const std::vector<std::string>& args);
