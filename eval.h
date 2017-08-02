#pragma once

#include "program.h"

#include <utility>

inline TensorValue eval(
        const Tensor& expr,
        const std::vector<Tensor>& args = {},
        const std::vector<const TensorValue *>& givens = {}) {
    return compile({expr}, args)(givens).front();
}
