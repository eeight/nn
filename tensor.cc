#include "tensor.h"
#include "ad.h"
#include "error.h"

#include <stdexcept>
#include <string>
#include <vector>
#include <functional>

namespace {

std::shared_ptr<Expr> maybeTile(
        const std::shared_ptr<Expr>& x, const Shape& shape, bool onlyScalar = false) {
    const auto& xShape = x->shape;
    if (xShape.isScalar()) {
        return x;
    } else if (onlyScalar) {
        return {};
    }

    if (xShape.dim() != shape.dim()) {
        return {};
    }

    std::vector<size_t> multiplierDim;

    for (size_t i = 0; i != shape.dim(); ++i) {
        if (shape(i) % xShape(i)) {
            return {};
        }
        multiplierDim.push_back(shape(i) / xShape(i));
    }

    return std::make_shared<Expr>(
            shape,
            Tile{Shape{std::move(multiplierDim)}},
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
    Shape shape = xExpr->shape;
    if (xExpr->shape != yExpr->shape) {
        if (auto tiled = maybeTile(xExpr, yExpr->shape)) {
            shape = yExpr->shape;
            xExpr = std::move(tiled);
        } else if (auto tiled = maybeTile(yExpr, xExpr->shape)) {
            shape = xExpr->shape;
            yExpr = std::move(tiled);
        } else {
            requireCompatible(false, "binary operator", xExpr->shape, yExpr->shape);
        }
    }
    return Tensor(std::make_shared<Expr>(
            std::move(shape), BinaryOp{op}, xExpr, yExpr));
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
    expr_(std::make_shared<Expr>(Shape{}, Const{x}))
{}

Tensor::~Tensor() = default;

const Shape& Tensor::shape() const {
    return expr_->shape;
}

Tensor Tensor::reshape(Shape newShape) const {
    requireCompatible(
            newShape.size() == shape().size(),
            "reshape",
            newShape,
            shape());
    if (shape() == newShape) {
        return *this;
    } else if (shape().t() == newShape) {
        return t();
    } else {
        return Tensor(std::make_shared<Expr>(newShape, Reshape{}, expr_));
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
        return konst->value.shape().isScalar() &&
            konst->value.toScalar() == 1.0f;
    } else {
        return false;
    }
}

void mutate(Tensor& tensor, const std::function<void (TensorValue&)>& mutator) {
    auto var = mpark::get_if<Var>(&tensor.unwrap()->op);
    if (!var) {
        throw std::runtime_error("Cannot mutate a non-variable");
    }
    mutator(var->value);
}

Tensor newPlaceholder(const Shape& shape) {
    return Tensor(std::make_shared<Expr>(shape, Placeholder{}));
}

Tensor newTensor(TensorValue init) {
    const auto shape = init.shape();
    return Tensor(std::make_shared<Expr>(
                shape, Var{std::move(init)}));
}

Tensor newConstTensor(TensorValue init) {
    const auto shape = init.shape();
    return Tensor(std::make_shared<Expr>(
                shape, Const{std::move(init)}));
}

Tensor operator +(const Tensor& x, const Tensor& y) {
    return binaryOpWithMatchingShapes(BinaryOperator::Plus, x, y);
}

Tensor operator -(const Tensor& x, const Tensor& y) {
    return binaryOpWithMatchingShapes(BinaryOperator::Plus, x, -y);
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

    if (!xExpr->shape.isMatrix() ||
            !yExpr->shape.isMatrix() ||
            xExpr->shape(1) != yExpr->shape(0)) {
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
        const size_t kRows = k.shape()(0);
        const size_t kCols = k.shape()(1);
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
    const size_t aRows = a.shape()(0);
    const size_t aCols = a.shape()(1);
    const size_t kRows = k.shape()(0);
    const size_t kCols = k.shape()(1);
    return Tensor(std::make_shared<Expr>(
                Shape{
                    aRows + conv.padTop + conv.padBottom + 1 - kRows,
                    aCols + conv.padLeft + conv.padRight + 1 - kCols},
                conv,
                a.unwrap(),
                k.unwrap()));
}

Tensor maxPool(const Tensor& a, Shape pool) {
    requireCompatible(
            a.shape().dim() == pool.dim(), "maxPool", a.shape(), pool);

    std::vector<size_t> dim;
    for (size_t i = 0; i != pool.dim(); ++i) {
        requireCompatible(
                a.shape()(i) % pool(i) == 0, "maxPool", a.shape(), pool);
        dim.push_back(a.shape()(i) / pool(i));
    }
    return Tensor(std::make_shared<Expr>(
                Shape{std::move(dim)},
                MaxPool{std::move(pool)},
                a.unwrap()));
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
        Shape{},
        HalfSumSquares{},
        tensor.unwrap()));
}

Tensor sum(const Tensor& tensor) {
    size_t size = tensor.shape().size();
    return (tensor.reshape({1, size}) *
        newConstTensor(TensorValue::ones({size, 1}))).reshape({});
}
