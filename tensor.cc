#include "tensor.h"
#include "ad.h"

#include <experimental/optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <functional>

namespace {

using Variables = std::vector<std::experimental::optional<Matrix>>;

Variables& variables() {
    static Variables variables;
    return variables;
}

class Const : public Expr {
public:
    explicit Const(Matrix value) :
        Expr(Shape(value)),
        value_(std::move(value))
    {}

    Matrix eval(Ad *) const override {
        return value_;
    }

    Matrix partial(
            const Expr*, const ValueGetter&, const Matrix&) const override {
        throw std::logic_error("Const::partial is not defined");
    }

private:
    Matrix value_;
};

class Var : public Expr {
public:
    Var(Shape shape, size_t id) :
        Expr(shape),
        id_(id)
    {}

    Matrix eval(Ad *ad) const override {
        const auto& var = variables().at(id_);
        if (!var) {
            throw std::runtime_error(
                    "Cannot read from uninitialized variable with id " +
                    std::to_string(id_));
        }
        if (ad) {
            ad->trace(this, {}, *var);
        }
        return *var;
    }

    Matrix partial(
            const Expr*, const ValueGetter&, const Matrix&) const override {
        throw std::logic_error("Var::partial is not defined");
    }

    size_t id() const { return id_; }

private:
    size_t id_;
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

    Matrix eval(Ad* ad) const override {
        Matrix result = Operator().eval(x_->eval(ad), y_->eval(ad));
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

    Matrix eval(Ad *ad) const override {
        Matrix result(1, 1);
        result.fill(std::pow(x_->eval(ad)(0, 0), y_));
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

class Reshape : public Expr {
public:
    Reshape(Shape shape, Shape originalShape, std::shared_ptr<Expr> x) :
        Expr(shape),
        originalShape_(originalShape),
        x_(std::move(x))
    {}

    Matrix eval(Ad *ad) const override {
        Matrix result = x_->eval(ad);
        result.reshape(shape().rows, shape().cols);
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

template <class Operator>
Tensor binaryOpWithMatchingShapes(const Tensor& x, const Tensor& y) {
    const auto& xExpr = unwrap(x);
    const auto& yExpr = unwrap(y);
    if (xExpr->shape() != yExpr->shape()) {
        throw std::runtime_error(
                std::string("Incompatible shapes in ") + Operator().name());
    }
    return Tensor(std::make_shared<BinaryOp<Operator>>(
            xExpr->shape(), xExpr, yExpr));
}

std::experimental::optional<Matrix>& extractVar(const Tensor& tensor) {
    auto varExpr = std::dynamic_pointer_cast<Var>(unwrap(tensor));
    if (!varExpr) {
        throw std::runtime_error("Expected a variable");
    }

    return variables().at(varExpr->id());
}

} // namespace

const std::shared_ptr<Expr>& unwrap(const Tensor& tensor) {
    return tensor.expr_;
}

Tensor::Tensor(std::shared_ptr<Expr> expr) :
    expr_(std::move(expr))
{}

Tensor::~Tensor() = default;

Matrix Tensor::eval(Ad *ad) const {
    return expr_->eval(ad);
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

Tensor& Tensor::operator =(const Matrix& matrix) {
    extractVar(*this) = matrix;
    return *this;
}

void Tensor::reset() {
    extractVar(*this) = {};
}

Tensor Tensor::reshape(Shape newShape) const {
    if (newShape.size() != shape().size()) {
        throw std::runtime_error(
                "Incompatible shape in Tensor::reshape");
    }
    return Tensor(std::make_shared<Reshape>(newShape, shape(), expr_));
}

Tensor newConstTensor(Matrix init) {
    return Tensor(std::make_shared<Const>(std::move(init)));
}

Tensor newTensor(size_t rows, size_t cols) {
    const size_t id = variables().size();
    variables().emplace_back();
    return Tensor(std::make_shared<Var>(Shape{rows, cols}, id));
}

Tensor newTensor(Matrix init) {
    const size_t id = variables().size();
    const Shape shape{init};
    variables().push_back(std::move(init));
    return Tensor(std::make_shared<Var>(shape, id));
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

Tensor operator *(const Tensor& x, const Tensor& y) {
    const auto& xExpr = unwrap(x);
    const auto& yExpr = unwrap(y);
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
