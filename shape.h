#pragma once

#include <cstddef>
#include <string>
#include <vector>
#include <cassert>

class Shape {
public:
    explicit Shape(std::vector<size_t> dim) :
        dim_(std::move(dim))
    {
        assert(size());
    }

    Shape(std::initializer_list<size_t> dim) :
        dim_{dim}
    {
        assert(size());
    }

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

