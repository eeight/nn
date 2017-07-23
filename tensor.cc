#include "tensor.h"
#include "ad.h"

#include <stdexcept>
#include <string>
#include <vector>
#include <functional>

namespace {

std::shared_ptr<Expr> maybeTile(
        const std::shared_ptr<Expr>& x, Shape shape, bool onlyScalar = false) {
    const auto xShape = x->shape;
    if (onlyScalar && !xShape.isScalar()) {
        return {};
    }
    if (shape.rows % xShape.rows || shape.cols % xShape.cols) {
        return {};
    }
    return std::make_shared<Expr>(
            shape,
            Tile{
                shape.rows / xShape.rows,
                shape.cols / xShape.cols,
                xShape},
            x);
}

std::shared_ptr<Expr> maybeTileScalar(
        const std::shared_ptr<Expr>& x, Shape shape) {
    return maybeTile(x, shape, /* onlyScalar = */ true);
}

Tensor binaryOpWithMatchingShapes(
        BinaryOperator op, const Tensor& x, const Tensor& y) {
    auto xExpr = x.unwrap();
    auto yExpr = y.unwrap();
    if (xExpr->shape != yExpr->shape) {
        if (auto tiled = maybeTile(xExpr, yExpr->shape)) {
            xExpr = std::move(tiled);
        } else if (auto tiled = maybeTile(yExpr, xExpr->shape)) {
            yExpr = std::move(tiled);
        } else {
            throw std::runtime_error(
                    std::string("Incompatible shapes in binary operator: (") +
                        std::to_string(xExpr->shape.rows) + ", " + std::to_string(xExpr->shape.cols) +
                        ") and (" +
                        std::to_string(yExpr->shape.rows) + ", " + std::to_string(yExpr->shape.cols) +
                        ")");
        }
    }
    return Tensor(std::make_shared<Expr>(
            xExpr->shape, BinaryOp{op}, xExpr, yExpr));
}

Matrix& extractVarForMutation(const Tensor& tensor, Shape shape) {
    if (shape != tensor.shape()) {
        throw std::runtime_error("Shape mismatch in assignment");
    }
    auto var = mpark::get_if<Var>(&tensor.unwrap()->op);
    if (!var) {
        throw std::runtime_error("Cannot mutate a non-variable");
    }
    auto value = mpark::get_if<Matrix>(&var->state);
    if (!value) {
        throw std::runtime_error(
                "Cannot mutate a placeholder variable " +
                mpark::get<std::string>(var->state));
    }
    return *value;
}

Matrix make11(float x) {
    Matrix result(1, 1);
    result.fill(x);
    return result;
}

Tensor makeShapePreservingMutator(const Tensor& x, Op mutator) {
    return Tensor(std::make_shared<Expr>(
                x.shape(), std::move(mutator), x.unwrap()));
}

} // namespace

Tensor::Tensor(std::shared_ptr<Expr> expr) :
    expr_(std::move(expr))
{}

Tensor::Tensor(float x) :
    expr_(std::make_shared<Expr>(Shape{1, 1}, Const{make11(x)}))
{}

Tensor::~Tensor() = default;

Shape Tensor::shape() const {
    return expr_->shape;
}

Tensor& Tensor::operator +=(const Matrix& matrix) {
    extractVarForMutation(*this, Shape{matrix}) += matrix;
    return *this;
}

Tensor& Tensor::operator =(Matrix matrix) {
    extractVarForMutation(*this, Shape{matrix}) = std::move(matrix);
    return *this;
}

Tensor Tensor::reshape(Shape newShape) const {
    if (newShape.size() != shape().size()) {
        throw std::runtime_error(
                "Incompatible shape in Tensor::reshape");
    }
    return Tensor(std::make_shared<Expr>(
                newShape, Reshape{newShape, shape()}, expr_));
}

Tensor Tensor::operator-() const {
    return makeShapePreservingMutator(*this, Negate{});
}

Tensor Tensor::t() const {
    return Tensor(std::make_shared<Expr>(shape().t(), Transpose{}, expr_));
}

Tensor newTensor(std::string name, size_t rows, size_t cols) {
    return newTensor(std::move(name), Shape{rows, cols});
}

Tensor newTensor(std::string name, Shape shape) {
    return Tensor(std::make_shared<Expr>(
                shape, Var{std::move(name)}));
}

Tensor newTensor(Matrix init) {
    const Shape shape{init};
    return Tensor(std::make_shared<Expr>(
                shape, Var{std::move(init)}));
}

Tensor newConstTensor(Matrix init) {
    const Shape shape{init};
    return Tensor(std::make_shared<Expr>(
                shape, Const{std::move(init)}));
}

Tensor operator +(const Tensor& x, const Tensor& y) {
    return binaryOpWithMatchingShapes(BinaryOperator::Plus, x, y);
}

Tensor operator -(const Tensor& x, const Tensor& y) {
    return binaryOpWithMatchingShapes(BinaryOperator::Minus, x, y);
}

Tensor operator %(const Tensor& x, const Tensor& y) {
    return binaryOpWithMatchingShapes(BinaryOperator::HadamardMul, x, y);
}

Tensor operator /(const Tensor& x, const Tensor& y) {
    return binaryOpWithMatchingShapes(BinaryOperator::HadamardDiv, x, y);
}

Tensor operator *(const Tensor& x, const Tensor& y) {
    const auto& xExpr = x.unwrap();
    const auto& yExpr = y.unwrap();
    if (xExpr->shape.cols != yExpr->shape.rows) {
        if (auto tiledX = maybeTileScalar(xExpr, yExpr->shape)) {
            return Tensor(std::move(tiledX)) % y;
        } else if (auto tiledY = maybeTileScalar(yExpr, xExpr->shape)) {
            return x % Tensor(std::move(tiledY));
        }
    }
    return Tensor(std::make_shared<Expr>(
            xExpr->shape * yExpr->shape,
            BinaryOp{BinaryOperator::Mul},
            xExpr,
            yExpr));
}

Tensor pow(const Tensor& x, float y) {
    if (y < 0) {
        throw std::runtime_error("Second argument to pow cannot be negative");
    }
    const auto& xExpr = x.unwrap();
    if (!xExpr->shape.isScalar()) {
        throw std::runtime_error(
                "pow can be applied only to scalars");
    }
    return Tensor(std::make_shared<Expr>(x.shape(), Pow{y}, xExpr));
}

Tensor exp(const Tensor& x) {
    return makeShapePreservingMutator(x, Exp{});
}

Tensor log(const Tensor& x) {
    return makeShapePreservingMutator(x, Log{});
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
