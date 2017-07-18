#include "tensor.h"

#include <experimental/optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <functional>

namespace t {

namespace {

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

} // namespace

class Expr {
public:
    explicit Expr(Shape shape) :
        shape_(shape)
    {}

    virtual ~Expr() = default;
    virtual Matrix eval() const = 0;
    Shape shape() const { return shape_; }

private:
    Shape shape_;
};

const std::shared_ptr<Expr>& unwrap(const Tensor& tensor) {
    return tensor.expr_;
}

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

    Matrix eval() const override {
        return value_;
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

    Matrix eval() const override {
        const auto& var = variables().at(id_);
        if (!var) {
            throw std::runtime_error(
                    "Cannot read from uninitialized variable with id " +
                    std::to_string(id_));
        }
        return *var;
    }

private:
    size_t id_;
};

using BinaryMatrixOp = std::function<Matrix (const Matrix&, const Matrix&)>;

class BinaryOp : public Expr {
public:
    BinaryOp(
            Shape shape,
            BinaryMatrixOp op,
            std::shared_ptr<Expr> x,
            std::shared_ptr<Expr> y) :
        Expr(shape),
        op_(std::move(op)),
        x_(std::move(x)),
        y_(std::move(y))
    {}

    Matrix eval() const override {
        return op_(x_->eval(), y_->eval());
    }

private:
    BinaryMatrixOp op_;
    std::shared_ptr<Expr> x_;
    std::shared_ptr<Expr> y_;
};

Tensor binaryOpWithMatchingShapes(
        const Tensor& x,
        const Tensor& y,
        BinaryMatrixOp op,
        const char* name) {
    const auto& xExpr = unwrap(x);
    const auto& yExpr = unwrap(y);
    if (xExpr->shape() != yExpr->shape()) {
        throw std::runtime_error(std::string("Incompatible shapes in ") + name);
    }
    return Tensor(std::make_shared<BinaryOp>(
            xExpr->shape(), std::move(op), xExpr, yExpr));
}

} // namespace

Tensor::Tensor(std::shared_ptr<Expr> expr) :
    expr_(std::move(expr))
{}

Tensor::~Tensor() = default;

Matrix Tensor::eval() const {
    return expr_->eval();
}

size_t Tensor::rows() const {
    return expr_->shape().rows;
}

size_t Tensor::cols() const {
    return expr_->shape().cols;
}

Tensor newConstTensor(Matrix init) {
    return Tensor(std::make_shared<Const>(std::move(init)));
}

Tensor newTensor(Matrix init) {
    const size_t id = variables().size();
    const Shape shape{init};
    variables().push_back(std::move(init));
    return Tensor(std::make_shared<Var>(shape, id));
}

Tensor operator +(const Tensor& x, const Tensor& y) {
    return binaryOpWithMatchingShapes(
            x,
            y,
            [](const Matrix& x, const Matrix& y) {
                return x + y;
            },
            "operator +");
}

Tensor operator -(const Tensor& x, const Tensor& y) {
    return binaryOpWithMatchingShapes(
            x,
            y,
            [](const Matrix& x, const Matrix& y) {
                return x - y;
            },
            "operator -");
}

Tensor operator %(const Tensor& x, const Tensor& y) {
    return binaryOpWithMatchingShapes(
            x,
            y,
            [](const Matrix& x, const Matrix& y) {
                return x % y;
            },
            "operator %");
}

Tensor operator *(const Tensor& x, const Tensor& y) {
    const auto& xExpr = unwrap(x);
    const auto& yExpr = unwrap(y);
    return Tensor(std::make_shared<BinaryOp>(
            xExpr->shape() * yExpr->shape(),
            [](const Matrix& x, const Matrix& y) {
                return x * y;
            },
            xExpr,
            yExpr));
}

} // namespace t
