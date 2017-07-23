#pragma once

#include "tensor.h"

std::vector<Tensor> diff(const Tensor& expr, const std::vector<Tensor>& vars);

