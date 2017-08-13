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

    auto begin() const { return dim_.begin(); }
    auto end() const { return dim_.end(); }

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

    Shape addFirstDim(size_t size) const;
    Shape dropFirstDim(size_t n = 1) const;

    Shape dropLastDim(size_t n = 1) const;
    Shape takeLastDim(size_t n) const;

    std::string toString() const;

private:
    std::vector<size_t> dim_;
};

