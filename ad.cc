#include "ad.h"
#include "expr.h"

void Ad::trace(
        const Expr* expr,
        const std::initializer_list<Expr* >& deps,
        const Matrix& value) {
    if (expressionToIndex_.count(expr)) {
        // Already computed.
        return;
    }
    const size_t index = expressions_.size();
    expressionToIndex_[expr] = index;
    expressions_.push_back(expr);
    expressionValues_.push_back(value);
    parents_.emplace_back();

    for (size_t i = 0; i != deps.size(); ++i) {
        parents_.at(expressionToIndex_.at(deps.begin()[i])).push_back(index);
    }
}

std::vector<Matrix> Ad::partial(const std::vector<t::Tensor>& vars) const {
    std::vector<Matrix> result(vars.size());
    std::vector<Matrix> partials(expressions_.size());

    partials.back() = arma::ones<Matrix>(1, 1);

    auto valueGetter = [this](const Expr* expr) {
        return expressionValues_.at(expressionToIndex_.at(expr));
    };

    std::unordered_map<const Expr *, size_t> expressionToVarIndex;
    for (size_t i = 0; i != vars.size(); ++i) {
        expressionToVarIndex.emplace(unwrap(vars[i]).get(), i);
    }

    for (size_t i = 1; i != expressions_.size(); ++i) {
        const size_t j = expressions_.size() - 1 - i;
        if (parents_.at(j).empty()) {
            continue;
        }
        const auto& parents = parents_.at(j);
        const auto* expr = expressions_.at(j);
        auto& partial = partials[j];

        const size_t firstParent = parents.front();
        partial = expressions_[firstParent]->partial(
                expr, valueGetter, partials[firstParent]);
        for (size_t parentIndex = 1; parentIndex != parents.size();
                ++parentIndex) {
            const auto parent = parents[parentIndex];
            partial += expressions_[parent]->partial(
                    expr, valueGetter, partials[parent]);
        }

        if (expressionToVarIndex.count(expr)) {
            result[expressionToVarIndex[expr]] = partial;
            expressionToVarIndex.erase(expr);
        }
    }

    if (!expressionToVarIndex.empty()) {
        throw std::runtime_error(
                "Given variable it not used in "
                "the expression being differentiated");
    }

    return result;
}
