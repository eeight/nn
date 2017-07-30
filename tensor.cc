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
    if (xShape.isScalar()) {
        return x;
    } else if (onlyScalar) {
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

Tensor Tensor::reshape(Shape newShape) const {
    if (newShape.size() != shape().size()) {
        throw std::runtime_error(
                "Incompatible shape in Tensor::reshape");
    }
    if (shape() == newShape) {
        return *this;
    } else if (shape().t() == newShape) {
        return t();
    } else {
        return Tensor(std::make_shared<Expr>(
                newShape, Reshape{newShape, shape()}, expr_));
    }
}

Tensor Tensor::operator-() const {
    return makeShapePreservingMutator(*this, Negate{});
}

Tensor Tensor::t() const {
    if (mpark::get_if<Transpose>(&expr_->op)) {
        // Transpose-transpose fusion.
        return Tensor(expr_->args.front());
    } else {
        return Tensor(std::make_shared<Expr>(shape().t(), Transpose{}, expr_));
    }
}

Tensor Tensor::r() const {
    if (mpark::get_if<Reverse>(&expr_->op)) {
        // Reverse-reverse fusion.
        return Tensor(expr_->args.front());
    } else {
        return Tensor(std::make_shared<Expr>(shape(), Reverse{}, expr_));
    }
}

bool Tensor::isConst1() const {
    if (const auto konst = mpark::get_if<Const>(&expr_->op)) {
        return Shape{konst->value}.isScalar() && konst->value(0, 0) == 1.0f;
    } else {
        return false;
    }
}

void mutate(Tensor& tensor, const std::function<void (Matrix&)>& mutator) {
    auto var = mpark::get_if<Var>(&tensor.unwrap()->op);
    if (!var) {
        throw std::runtime_error("Cannot mutate a non-variable");
    }
    mutator(var->value);
}

Tensor newTensor(size_t rows, size_t cols) {
    return newTensor(Shape{rows, cols});
}

Tensor newTensor(Shape shape) {
    return Tensor(std::make_shared<Expr>(shape, Placeholder{}));
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
    if (x.isConst1()) {
        return y;
    } else if (y.isConst1()) {
        return x;
    } else {
        return binaryOpWithMatchingShapes(BinaryOperator::HadamardMul, x, y);
    }
}

Tensor operator /(const Tensor& x, const Tensor& y) {
    if (y.isConst1()) {
        return x;
    } else {
        return binaryOpWithMatchingShapes(BinaryOperator::HadamardDiv, x, y);
    }
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

Tensor conv2d(const Tensor& a, const Tensor& k, bool sameSize) {
    if (sameSize) {
        const size_t kRows = k.shape().rows;
        const size_t kCols = k.shape().cols;
        return conv2d(a, k, {
                /* padTop = */ kRows / 2,
                /* padBottom = */ (kRows - 1) / 2,
                /* padLeft = */ kCols / 2,
                /* padRight = */ (kCols - 1) / 2});
    } else {
        return conv2d(a, k, {0, 0, 0, 0});
    }
}

Tensor conv2d(
        const Tensor& a,
        const Tensor& k,
        const Conv2D& conv) {
    return Tensor(std::make_shared<Expr>(
                Shape{
                    a.shape().rows + conv.padTop + conv.padBottom + 1 -
                        k.shape().rows,
                    a.shape().cols + conv.padLeft + conv.padRight + 1 -
                        k.shape().cols},
                conv,
                a.unwrap(),
                k.unwrap()));
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

Tensor sigmoid(const Tensor& x) {
    return makeShapePreservingMutator(x, Sigmoid{});
}

Tensor halfSumSquares(const Tensor& tensor) {
    return Tensor(std::make_shared<Expr>(
        Shape{1, 1},
        HalfSumSquares{},
        tensor.unwrap()));
}

Tensor sum(const Tensor& tensor) {
    size_t size = tensor.shape().size();
    return tensor.reshape({1, size}) *
        newConstTensor(arma::ones<Matrix>(size, 1));
}
