#pragma once

#include "program.h"

#include <utility>

inline Matrix eval(
        const Tensor& expr,
        const std::vector<Tensor>& args = {},
        const std::vector<const Matrix *>& givens = {}) {
    return compile({expr}, args)(givens).front();
}
