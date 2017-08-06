#pragma once

#include "tensor.h"

Tensor quadraticLoss(const Tensor& out, const Tensor& target);
Tensor crossEntropyLoss(const Tensor& out, const Tensor& target);
