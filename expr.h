#pragma once

#include "types.h"

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

    size_t rows;
    size_t cols;
};

class Ad;

class Expr {
public:
    explicit Expr(Shape shape) :
        shape_(shape)
    {}

    using ValueGetter = std::function<Matrix (const Expr* expr)>;

    virtual ~Expr() = default;
    virtual Matrix eval(Ad *ad) const = 0;
    virtual Matrix partial(
            const Expr* subexpr,
            const ValueGetter& valueGetter,
            const Matrix& selfPartial) const = 0;
    Shape shape() const { return shape_; }

private:
    Shape shape_;
};
