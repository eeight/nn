#pragma once

#include "types.h"
#include "memory.h"

class Loss {
public:
    virtual float loss(const Col& out, const Col& target) const = 0;
    virtual Matrix delta(
            const Matrix& out, const Matrix& target, const Matrix& z) const = 0;
    virtual ~Loss() = default;
};

std::unique_ptr<Loss> quadraticLoss();
std::unique_ptr<Loss> crossEntropyLoss();
