#pragma once

#include "tensor.h"

#include <unordered_map>
#include <vector>

std::vector<Tensor> diff(const Tensor& expr, const std::vector<Tensor>& vars);

class Ad {
public:
    void trace(
            const Expr* expr,
            const std::initializer_list<Expr* >& deps,
            std::shared_ptr<Matrix> value);

    std::vector<Matrix> partial(const std::vector<Tensor>& vars) const;

private:
    // List of sub-expression computed during the evaluation
    std::vector<const Expr *> expressions_;
    // Values of sub-expressions.
    std::vector<std::shared_ptr<Matrix>> expressionValues_;

    // For each expression list of expressions that depend
    // on it.
    std::vector<std::vector<size_t>> parents_;

    std::unordered_map<const Expr *, size_t> expressionToIndex_;
};
