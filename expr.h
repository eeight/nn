#pragma once

#include "types.h"
#include "shape.h"

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
