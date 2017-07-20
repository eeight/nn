#include "tensor.h"
#include "ad.h"

#include <stdexcept>
#include <string>
#include <vector>
#include <functional>

namespace {

class Const : public Expr {
public:
    explicit Const(Matrix value) :
        Expr(Shape(value)),
        value_(std::make_shared<Matrix>(std::move(value)))
    {}

    std::shared_ptr<Matrix> eval(Ad *ad) const override {
        if (ad) {
            ad->trace(this, {}, value_);
        }
        return value_;
    }

    Matrix partial(
            const Expr*, const ValueGetter&, const Matrix&) const override {
        throw std::logic_error("Const::partial is not defined");
    }

    const Matrix& value() const { return *value_; }

private:
    std::shared_ptr<Matrix> value_;
};

class Var : public Expr {
public:
    Var(Shape shape, std::shared_ptr<Matrix> value) :
        Expr(shape),
        value_(std::move(value))
    {}

    std::shared_ptr<Matrix> eval(Ad *ad) const override {
        if (!value_) {
            throw std::runtime_error(
                    "Cannot read from uninitialized variable");
        }
        if (ad) {
            ad->trace(this, {}, value_);
        }
        return value_;
    }

    Matrix partial(
            const Expr*, const ValueGetter&, const Matrix&) const override {
        throw std::logic_error("Var::partial is not defined");
    }

    std::shared_ptr<Matrix>& value() { return value_; }

private:
    std::shared_ptr<Matrix> value_;
};

class Tiled : public Expr {
public:
    Tiled(std::shared_ptr<Expr> x, size_t repeatRows, size_t repeatCols) :
        Expr(Shape{
                x->shape().rows * repeatRows,
                x->shape().cols * repeatCols}),
        x_(std::move(x)),
        repeatRows_(repeatRows),
        repeatCols_(repeatCols),
        originalShape_(x_->shape())
    {}

    std::shared_ptr<Matrix> eval(Ad* ad) const override {
        auto result = std::make_shared<Matrix>(
                repmat(*x_->eval(ad), repeatRows_, repeatCols_));
        if (ad) {
            ad->trace(this, {x_.get()}, result);
        }
        return result;
    }

    Matrix partial(
            const Expr* expr,
            const ValueGetter&,
            const Matrix& selfPartial) const override {
        if (expr != x_.get()) {
            throw std::logic_error("Unexpected expr in Tiled::partial");
        }

        Matrix result(
                originalShape_.rows, originalShape_.cols, arma::fill::zeros);
        for (size_t i = 0; i != repeatRows_; ++i) {
            const size_t beginRow = i * originalShape_.rows;
            for (size_t j = 0; j != repeatCols_; ++j) {
                const size_t beginCol = j * originalShape_.cols;
                result += selfPartial.submat(
                        beginRow,
                        beginCol,
                        // Subtract one because the ranges are inclusive here.
                        beginRow + originalShape_.rows - 1,
                        beginCol + originalShape_.cols - 1);
            }
        }
        return result;
    }

private:
    std::shared_ptr<Expr> x_;
    size_t repeatRows_;
    size_t repeatCols_;
    Shape originalShape_;
};

struct OperatorPlus {
    const char *name() const {
        return "operator +";
    }

    Matrix eval(const Matrix& x, const Matrix& y) const {
        return x + y;
    }

    Matrix partial(
            const Expr*,
            const Expr*,
            const Expr*,
            const Expr::ValueGetter&,
            const Matrix& selfPartial) const {
        return selfPartial;
    }
};

struct OperatorMinus {
    const char *name() const {
        return "operator -";
    }

    Matrix eval(const Matrix& x, const Matrix& y) const {
        return x - y;
    }

    Matrix partial(
            const Expr* x,
            const Expr*,
            const Expr* p,
            const Expr::ValueGetter&,
            const Matrix& selfPartial) const {
        if (p == x) {
            return selfPartial;
        } else {
            return -selfPartial;
        }
    }
};

struct OperatorHadamardProduct {
    const char *name() const {
        return "operator %";
    }

    Matrix eval(const Matrix& x, const Matrix& y) const {
        return x % y;
    }

    Matrix partial(
            const Expr* x,
            const Expr* y,
            const Expr* p,
            const Expr::ValueGetter& value,
            const Matrix& selfPartial) const {
        if (p == x) {
            return value(y) % selfPartial;
        } else {
            return value(x) % selfPartial;
        }
    }
};

struct OperatorHadamardDivision {
    const char *name() const {
        return "operator /";
    }

    Matrix eval(const Matrix& x, const Matrix& y) const {
        return x / y;
    }

    Matrix partial(
            const Expr* x,
            const Expr* y,
            const Expr* p,
            const Expr::ValueGetter& value,
            const Matrix& selfPartial) const {
        if (p == x) {
            return 1.0 / value(y) % selfPartial;
        } else {
            const auto yValue = value(y);
            return -value(x) / (yValue % yValue) % selfPartial;
        }
    }
};

struct OperatorMul {
    const char *name() const {
        return "operator *";
    }

    Matrix eval(const Matrix& x, const Matrix& y) const {
        return x * y;
    }

    Matrix partial(
            const Expr* x,
            const Expr* y,
            const Expr* p,
            const Expr::ValueGetter& value,
            const Matrix& selfPartial) const {
        if (p == x) {
            return selfPartial * value(y).t();
        } else {
            return value(x).t() * selfPartial;
        }
    }
};

template <class Operator>
class BinaryOp : public Expr {
public:
    BinaryOp(
            Shape shape,
            std::shared_ptr<Expr> x,
            std::shared_ptr<Expr> y) :
        Expr(shape),
        x_(std::move(x)),
        y_(std::move(y))
    {}

    std::shared_ptr<Matrix> eval(Ad* ad) const override {
        auto result = std::make_shared<Matrix>(
                Operator().eval(*x_->eval(ad), *y_->eval(ad)));
        if (ad) {
            ad->trace(this, {x_.get(), y_.get()}, result);
        }
        return result;
    }

    Matrix partial(
            const Expr* expr,
            const ValueGetter& value,
            const Matrix& selfPartial) const override {
        if (expr != x_.get() && expr != y_.get()) {
            throw std::logic_error("Unexpected expr in BinaryOp::partial");
        }
        auto result = Operator().partial(
                x_.get(), y_.get(), expr, value, selfPartial);
        return result;
    }

private:
    std::shared_ptr<Expr> x_;
    std::shared_ptr<Expr> y_;
};

class Pow : public Expr {
public:
    Pow(std::shared_ptr<Expr> x, float y) :
        Expr({1, 1}),
        x_(std::move(x)),
        y_(y)
    {}

    std::shared_ptr<Matrix> eval(Ad *ad) const override {
        auto result = std::make_shared<Matrix>(1, 1);
        result->fill(std::pow((*x_->eval(ad))(0, 0), y_));
        if (ad) {
            ad->trace(this, {x_.get()}, result);
        }
        return result;
    }

    Matrix partial(
            const Expr* expr,
            const ValueGetter& value,
            const Matrix& selfPartial) const override {
        if (expr != x_.get()) {
            throw std::logic_error("Unexpected expr in Pow::partial");
        }
        return selfPartial * (y_ * std::pow(value(x_.get())(0, 0), y_ - 1));
    }

private:
    std::shared_ptr<Expr> x_;
    float y_;
};

class Exp : public Expr {
public:
    Exp(std::shared_ptr<Expr> x) :
        Expr(x->shape()),
        x_(std::move(x))
    {}

    std::shared_ptr<Matrix> eval(Ad *ad) const override {
        auto result = std::make_shared<Matrix>(exp(*x_->eval(ad)));
        if (ad) {
            ad->trace(this, {x_.get()}, result);
        }
        return result;
    }

    Matrix partial(
            const Expr* expr,
            const ValueGetter& value,
            const Matrix& selfPartial) const override {
        if (expr != x_.get()) {
            throw std::logic_error("Unexpected expr in Exp::partial");
        }
        return exp(value(expr)) % selfPartial;
    }

private:
    std::shared_ptr<Expr> x_;
};

class Log : public Expr {
public:
    Log(std::shared_ptr<Expr> x) :
        Expr(x->shape()),
        x_(std::move(x))
    {}

    std::shared_ptr<Matrix> eval(Ad *ad) const override {
        auto result = std::make_shared<Matrix>(log(*x_->eval(ad)));
        if (ad) {
            ad->trace(this, {x_.get()}, result);
        }
        return result;
    }

    Matrix partial(
            const Expr* expr,
            const ValueGetter& value,
            const Matrix& selfPartial) const override {
        if (expr != x_.get()) {
            throw std::logic_error("Unexpected expr in Exp::partial");
        }
        return 1.0f / value(expr) % selfPartial;
    }

private:
    std::shared_ptr<Expr> x_;
};

class Reshape : public Expr {
public:
    Reshape(Shape shape, Shape originalShape, std::shared_ptr<Expr> x) :
        Expr(shape),
        originalShape_(originalShape),
        x_(std::move(x))
    {}

    std::shared_ptr<Matrix> eval(Ad *ad) const override {
        auto result = std::make_shared<Matrix>(*x_->eval(ad));
        result->reshape(shape().rows, shape().cols);
        if (ad) {
            ad->trace(this, {x_.get()}, result);
        }
        return result;
    }

    Matrix partial(
            const Expr* expr,
            const ValueGetter&,
            const Matrix& selfPartial) const override {
        if (expr != x_.get()) {
            throw std::logic_error("Unexpected expr in Reshape::partial");
        }
        auto result = selfPartial;
        result.reshape(originalShape_.rows, originalShape_.cols);
        return result;
    }

private:
    Shape originalShape_;
    std::shared_ptr<Expr> x_;
};

class Negate : public Expr {
public:
    Negate(std::shared_ptr<Expr> x) :
        Expr(x->shape()),
        x_(std::move(x))
    {}

    std::shared_ptr<Matrix> eval(Ad *ad) const override {
        auto result = std::make_shared<Matrix>(-*x_->eval(ad));
        if (ad) {
            ad->trace(this, {x_.get()}, result);
        }
        return result;
    }

    Matrix partial(
            const Expr* expr,
            const ValueGetter&,
            const Matrix& selfPartial) const override {
        if (expr != x_.get()) {
            throw std::logic_error("Unexpected expr in Negate::partial");
        }
        return -selfPartial;
    }

private:
    std::shared_ptr<Expr> x_;
};

std::shared_ptr<Tiled> maybeTile(
        const std::shared_ptr<Expr>& x, Shape shape, bool onlyScalar = false) {
    const auto xShape = x->shape();
    if (onlyScalar && xShape != Shape{1, 1}) {
        return {};
    }
    if (shape.rows % xShape.rows || shape.cols % xShape.cols) {
        return {};
    }
    return std::make_shared<Tiled>(
            x, shape.rows / xShape.rows, shape.cols / xShape.cols);
}

std::shared_ptr<Tiled> maybeTileScalar(
        const std::shared_ptr<Expr>& x, Shape shape) {
    return maybeTile(x, shape, /* onlyScalar = */ true);
}

template <class Operator>
Tensor binaryOpWithMatchingShapes(const Tensor& x, const Tensor& y) {
    auto xExpr = unwrap(x);
    auto yExpr = unwrap(y);
    if (xExpr->shape() != yExpr->shape()) {
        if (auto tiled = maybeTile(xExpr, yExpr->shape())) {
            xExpr = std::move(tiled);
        } else if (auto tiled = maybeTile(yExpr, xExpr->shape())) {
            yExpr = std::move(tiled);
        } else {
            throw std::runtime_error(
                    std::string("Incompatible shapes in ") + Operator().name());
        }
    }
    return Tensor(std::make_shared<BinaryOp<Operator>>(
            xExpr->shape(), xExpr, yExpr));
}

std::shared_ptr<Matrix>& extractVar(const Tensor& tensor) {
    auto varExpr = std::dynamic_pointer_cast<Var>(unwrap(tensor));
    if (!varExpr) {
        throw std::runtime_error("Expected a variable");
    }

    return varExpr->value();
}

Matrix make11(float x) {
    Matrix result(1, 1);
    result.fill(x);
    return result;
}

} // namespace

const std::shared_ptr<Expr>& unwrap(const Tensor& tensor) {
    return tensor.expr_;
}

Tensor::Tensor(std::shared_ptr<Expr> expr) :
    expr_(std::move(expr))
{}

Tensor::Tensor(float x) :
    expr_(std::make_shared<Const>(make11(x)))
{}

Tensor::~Tensor() = default;

Matrix Tensor::eval(Ad *ad) const {
    return *expr_->eval(ad);
}

Shape Tensor::shape() const {
    return expr_->shape();
}

Tensor& Tensor::operator +=(const Matrix& matrix) {
    auto& var = extractVar(*this);
    if (!var) {
        throw std::runtime_error(
                "Cannot apply operator += to uninitialized variable");
    }

    *var += matrix;

    return *this;
}

Tensor& Tensor::operator =(Matrix matrix) {
    if (Shape{matrix} != shape()) {
        throw std::runtime_error("Shape mismatch in tensor assignment");
    }
    extractVar(*this) = std::make_shared<Matrix>(std::move(matrix));
    return *this;
}

void Tensor::reset() {
    extractVar(*this).reset();
}

Tensor Tensor::reshape(Shape newShape) const {
    if (newShape.size() != shape().size()) {
        throw std::runtime_error(
                "Incompatible shape in Tensor::reshape");
    }
    return Tensor(std::make_shared<Reshape>(newShape, shape(), expr_));
}

Tensor Tensor::operator-() const {
    return Tensor(std::make_shared<Negate>(expr_));
}

Tensor newConstTensor(Matrix init) {
    return Tensor(std::make_shared<Const>(std::move(init)));
}

Tensor newTensor(size_t rows, size_t cols) {
    return newTensor(Shape{rows, cols});
}

Tensor newTensor(Shape shape) {
    return Tensor(std::make_shared<Var>(shape, nullptr));
}

Tensor newTensor(Matrix init) {
    const Shape shape{init};
    return Tensor(std::make_shared<Var>(
                shape, std::make_shared<Matrix>(std::move(init))));
}

Tensor operator +(const Tensor& x, const Tensor& y) {
    return binaryOpWithMatchingShapes<OperatorPlus>(x, y);
}

Tensor operator -(const Tensor& x, const Tensor& y) {
    return binaryOpWithMatchingShapes<OperatorMinus>(x, y);
}

Tensor operator %(const Tensor& x, const Tensor& y) {
    return binaryOpWithMatchingShapes<OperatorHadamardProduct>(x, y);
}

Tensor operator /(const Tensor& x, const Tensor& y) {
    return binaryOpWithMatchingShapes<OperatorHadamardDivision>(x, y);
}

Tensor operator *(const Tensor& x, const Tensor& y) {
    const auto& xExpr = unwrap(x);
    const auto& yExpr = unwrap(y);
    if (xExpr->shape().cols != yExpr->shape().rows) {
        if (auto tiled = maybeTileScalar(xExpr, yExpr->shape())) {
            return Tensor(std::make_shared<BinaryOp<OperatorHadamardProduct>>(
                        yExpr->shape(),
                        tiled,
                        yExpr));
        } else if (auto tiled = maybeTileScalar(yExpr, xExpr->shape())) {
            return Tensor(std::make_shared<BinaryOp<OperatorHadamardProduct>>(
                        xExpr->shape(),
                        xExpr,
                        tiled));
        }
    }
    return Tensor(std::make_shared<BinaryOp<OperatorMul>>(
            xExpr->shape() * yExpr->shape(), xExpr, yExpr));
}

Tensor pow(const Tensor& x, float y) {
    if (y < 0) {
        throw std::runtime_error("Second argument to pow cannot be negative");
    }
    const auto& xExpr = unwrap(x);
    if (xExpr->shape() != Shape{1, 1}) {
        throw std::runtime_error(
                "pow can be applied only to tensors with shape(1, 1)");
    }
    return Tensor(std::make_shared<Pow>(xExpr, y));
}

Tensor exp(const Tensor& x) {
    return Tensor(std::make_shared<Exp>(unwrap(x)));
}

Tensor log(const Tensor& x) {
    return Tensor(std::make_shared<Log>(unwrap(x)));
}

std::vector<Matrix> diff(const Tensor& expr, const std::vector<Tensor>& vars) {
    Ad ad;
    expr.eval(&ad);
    return ad.partial(vars);
}

Tensor sumSquares(const Tensor& tensor) {
    size_t size = tensor.shape().size();
    return tensor.reshape({1, size}) * tensor.reshape({size, 1});
}

Tensor sum(const Tensor& tensor) {
    size_t size = tensor.shape().size();
    return tensor.reshape({1, size}) *
        newConstTensor(arma::ones<Matrix>(size, 1));
}
