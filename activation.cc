#include "activation.h"

Tensor sigmoid(const Tensor& x) {
    return 1.0f / (1.0f + exp(-x));
}
