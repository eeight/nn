#pragma once

#include "types.h"
#include "shape.h"

class Ad;

class Expr {
public:
    explicit Expr(Shape shape) :
        shape_(shape)
    {}

    using ValueGetter = std::function<const Matrix& (const Expr* expr)>;

    virtual ~Expr() = default;
    virtual std::shared_ptr<Matrix> eval(Ad *ad) const = 0;
    virtual Matrix partial(
            const Expr* subexpr,
            const ValueGetter& valueGetter,
            const Matrix& selfPartial) const = 0;
    Shape shape() const { return shape_; }

private:
    Shape shape_;
};
