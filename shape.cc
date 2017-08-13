#include "shape.h"
#include "error.h"

#include <functional>
#include <numeric>
#include <stdexcept>

Shape Shape::operator *(const Shape& other) const {
    requireCompatible(
            isMatrix() && other.isMatrix() && dim_[1] == other(0),
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

Shape Shape::addFirstDim(size_t size) const {
    std::vector<size_t> dim;
    dim.reserve(dim_.size() + 1);
    dim.push_back(size);
    dim.insert(dim.end(), dim_.begin(), dim_.end());
    return Shape{std::move(dim)};
}

Shape Shape::dropFirstDim(size_t n) const {
    if (dim() < n) {
        throw std::logic_error(
                "Cannot apply dropFirstDim(" + std::to_string(n) + ") to shape " + toString());
    }
    return Shape{std::vector<size_t>{dim_.begin() + n, dim_.end()}};
}

Shape Shape::dropLastDim(size_t n) const {
    if (dim() < n) {
        throw std::logic_error(
                "Cannot apply dropLastDim(" + std::to_string(n) + ") to shape " + toString());
    }
    return Shape{std::vector<size_t> (dim_.begin(), dim_.end() - n)};
}

Shape Shape::takeLastDim(size_t n) const {
    if (dim() < n) {
        throw std::logic_error(
                "Cannot apply takeLastDim(" + std::to_string(n) + ") to shape " + toString());
    }
    return dropFirstDim(dim() - n);
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
