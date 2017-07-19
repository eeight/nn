#pragma once

#include "types.h"

#include <cstddef>

struct Shape {
    explicit Shape(const Matrix& matrix) :
        rows(matrix.n_rows), cols(matrix.n_cols)
    {}

    Shape(size_t rows, size_t cols) :
        rows(rows), cols(cols)
    {}

    bool operator ==(Shape other) const {
        return rows == other.rows && cols == other.cols;
    }

    bool operator !=(Shape other) const {
        return !(*this == other);
    }

    Shape operator *(Shape other) const {
        if (cols != other.rows) {
            throw std::runtime_error(
                    "Incompatible shapes for matrix multiplication");
        }

        return {rows, other.cols};
    }

    size_t size() const {
        return cols * rows;
    }

    bool isScalar() const {
        return rows == 1 && cols == 1;
    }

    size_t rows;
    size_t cols;
};

