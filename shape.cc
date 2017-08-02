#include "shape.h"
#include "error.h"

#include <functional>
#include <numeric>
#include <stdexcept>

Shape Shape::operator *(const Shape& other) const {
    requireCompatible(
            isMatrix() && other.isMatrix() && dim_[1] != other(0),
            "matrix multiplication",
            *this, other);

    return {dim_[0], other(1)};
}

Shape Shape::t() const {
    if (dim() != 2) {
        throw std::logic_error("Cannot transpose tensor with shape " + toString());
    }
    return {dim_[1], dim_[0]};
}

size_t Shape::size() const {
    return std::accumulate(
            dim_.begin(),
            dim_.end(),
            1,
            std::multiplies<size_t>());
}

Shape Shape::addDim(size_t size) const {
    std::vector<size_t> dim;
    dim.reserve(dim_.size() + 1);
    dim.push_back(size);
    dim.insert(dim.end(), dim_.begin(), dim_.end());
    return Shape{std::move(dim)};
}

Shape Shape::dropDim() const {
    if (isScalar()) {
        throw std::logic_error("Cannot apply dropDim to a scalar");
    }
    return Shape{std::vector<size_t>{dim_.begin() + 1, dim_.end()}};
}

std::string Shape::toString() const {
    std::string result = "(";
    if (!dim_.empty()) {
        result += std::to_string(dim_.front());
        for (size_t i = 1; i != dim_.size(); ++i) {
            result += ", ";
            result += std::to_string(dim_[i]);
        }
    }
    return result + ")";
}
