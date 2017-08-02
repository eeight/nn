#pragma once

#include "shape.h"

void requireCompatible(
        bool condition,
        const char* context,
        const Shape& x,
        const Shape& y);
