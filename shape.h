#pragma once

#include "types.h"

#include <cstddef>
#include <string>

class Shape {
public:
    explicit Shape(std::vector<size_t> dim) :
        dim_(std::move(dim))
    {}

    explicit Shape(const Matrix& matrix) :
        Shape({matrix.n_rows, matrix.n_cols})
    {}

    Shape(std::initializer_list<size_t> dim) :
        dim_{dim}
    {}

    size_t dim() const { return dim_.size(); }

    size_t operator()(size_t i) const { return dim_.at(i); }

    bool operator ==(const Shape& other) const {
        return dim_ == other.dim_;
    }

    bool operator !=(const Shape& other) const {
        return !(*this == other);
    }

    Shape operator *(const Shape& other) const;

    Shape t() const;
    size_t size() const;

    bool isScalar() const {
        return dim_.empty();
    }

    bool isVector() const {
        return dim_.size() == 1;
    }

    bool isMatrix() const {
        return dim_.size() == 2;
    }

    bool isCube() const {
        return dim_.size() == 3;
    }

    Shape addDim(size_t size) const;
    Shape dropDim() const;

    std::string toString() const;

private:
    std::vector<size_t> dim_;
};

