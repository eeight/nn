#pragma once

#include "tensor.h"

#include <unordered_map>
#include <vector>

class Ad {
public:
    void trace(
            const Expr* expr,
            const std::initializer_list<Expr* >& deps,
            const Matrix& value);

    std::vector<Matrix> partial(const std::vector<t::Tensor>& vars) const;

private:
    // List of sub-expression computed during the evaluation
    std::vector<const Expr *> expressions_;
    // Values of sub-expressions.
    std::vector<Matrix> expressionValues_;

    // For each expression list of expressions that depend
    // on it.
    std::vector<std::vector<size_t>> parents_;

    std::unordered_map<const Expr *, size_t> expressionToIndex_;
};
