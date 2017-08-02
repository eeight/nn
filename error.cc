#include "error.h"

#include <stdexcept>
#include <string>

void requireCompatible(
        bool condition,
        const char* context,
        const Shape& x,
        const Shape& y) {
    if (!condition) {
        throw std::logic_error(
                "Incompatible shapes in " + std::string(context) +  ": " +
                x.toString() + " and " + y.toString());
    }
}
