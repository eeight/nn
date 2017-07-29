#include "ad.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

struct PartialDiff {
    Tensor operator()(const Tile& tile) const {
        return Tensor(std::make_shared<Expr>(
            tile.originalShape,
            Untile{
                tile.repeatRows,
                tile.repeatCols,
                tile.originalShape},
            selfPartial.unwrap()));
    }

    Tensor operator()(const Untile& untile) const {
        return Tensor(std::make_shared<Expr>(
            Shape{
                untile.originalShape.rows * untile.repeatRows,
                untile.originalShape.cols * untile.repeatCols},
            Tile{
                untile.repeatRows,
                untile.repeatCols,
                untile.originalShape},
            selfPartial.unwrap()));
    }

    Tensor operator()(const BinaryOp& binaryOp) const {
        switch (binaryOp.op) {
            case BinaryOperator::Plus:
                return selfPartial;

            case BinaryOperator::Minus:
                if (arg == 0) {
                    return selfPartial;
                } else {
                    return -selfPartial;
                }

            case BinaryOperator::Mul:
                if (arg == 0) {
                    return selfPartial * y().t();
                } else {
                    return x().t() * selfPartial;

                }
            case BinaryOperator::HadamardMul:
                if (arg == 0) {
                    return y() % selfPartial;
                } else {
                    return x() % selfPartial;
                }

            case BinaryOperator::HadamardDiv:
                if (arg == 0) {
                    return 1.0 / y() % selfPartial;
                } else {
                    return -x() / (y() % y()) % selfPartial;
                }
        }
        abort();
    }

    Tensor operator()(const Conv2D& conv) const {
        const auto& a = x();
        const auto& k = y();

        const size_t kRows = k.shape().rows;
        const size_t kCols = k.shape().cols;

        if (arg == 0) {
            return conv2d(selfPartial, k.r(), {
                    kCols - 1 - conv.padTop,
                    kCols - 1 - conv.padBottom,
                    kRows - 1 - conv.padLeft,
                    kRows - 1 - conv.padRight});
        } else {
            return conv2d(a, selfPartial, conv);
        }
    }

    Tensor operator()(const Pow& p) const {
        if (p.y == 1) {
            return selfPartial;
        } else if (p.y == 2) {
            return selfPartial % p.y % x();
        } else {
            return selfPartial % p.y % pow(x(), p.y - 1);
        }
    }

    Tensor operator()(const Exp&) const {
        return self % selfPartial;
    }

    Tensor operator()(const Log&) const {
        return 1.0 / x() % selfPartial;
    }

    Tensor operator()(const Negate&) const {
        return -selfPartial;
    }

    Tensor operator()(const Transpose&) const {
        return selfPartial.t();
    }

    Tensor operator()(const Reverse&) const {
        return selfPartial.r();
    }

    Tensor operator()(const Reshape& reshape) const {
        return selfPartial.reshape(reshape.originalShape);
    }

    Tensor operator()(const Sigmoid&) const {
        return self % (1.0f - self) % selfPartial;
    }

    Tensor operator()(const HalfSumSquares&) const {
        return selfPartial * x();
    }

    template <class T>
    Tensor operator()(const T&) const {
        throw std::logic_error("Unexpected op in PartialDiff");
    }

    Tensor x() const { return Tensor(args.at(0)); }
    Tensor y() const { return Tensor(args.at(1)); }

    const Tensor& self;
    const Tensor& selfPartial;
    const std::vector<std::shared_ptr<Expr>>& args;
    size_t arg;
};

class Ad {
public:
    explicit Ad(const Tensor& tensor) {
        if (!tensor.shape().isScalar()) {
            throw std::logic_error("Cannot differentiate a non-scalar value");
        }
        trace(tensor.unwrap());
        partial_.emplace(
                tensor.unwrap().get(),
                newConstTensor(arma::ones<Matrix>(1, 1)));
    }

    void trace(const std::shared_ptr<Expr>& expr) {
        if (traced_.count(expr.get())) {
            return;
        }
        traced_.insert(expr.get());

        for (const auto& arg: expr->args) {
            reverseDeps_[arg.get()].push_back(Tensor(expr));
        }
        for (const auto& arg: expr->args) {
            trace(arg);
        }
    }

    std::vector<Tensor> partial(const std::vector<Tensor>& vars) {
        std::vector<Tensor> result;
        for (const auto& var: vars) {
            result.push_back(partial(var.unwrap().get()));
        }

        return result;
    }

private:
    Tensor partial(const Expr* expr) {
        {
            const auto iter = partial_.find(expr);
            if (iter != partial_.end()) {
                return iter->second;
            }
        }

        const auto iter = reverseDeps_.find(expr);
        if (iter == reverseDeps_.end()) {
            // Result does not even depend on this expression.

            auto result = newConstTensor(
                    arma::zeros<Matrix>(
                        expr->shape.rows,
                        expr->shape.cols));
            partial_.emplace(expr, result);
            return result;
        }

        const auto& parents = iter->second;
        Tensor result = partial(parents.front(), expr);
        for (size_t i = 1; i != parents.size(); ++i) {
            const auto& parent = parents[i];
            result = result + partial(parent, expr);
        }
        partial_.emplace(expr, result);
        return result;
    }

    Tensor partial(
            const Tensor& tensor, const Expr* arg) {
        const auto& selfPartial = partial(tensor.unwrap().get());
        const auto& expr = *tensor.unwrap();
        for (size_t i = 0; i != expr.args.size(); ++i) {
            if (expr.args[i].get() == arg) {
                return mpark::visit(
                        PartialDiff{tensor, selfPartial, expr.args, i}, expr.op);
            }
        }

        throw std::logic_error("Ad::partial: arg is not in expr.args");
    }

    std::unordered_map<const Expr *, std::vector<Tensor>> reverseDeps_;
    std::unordered_set<const Expr*> traced_;
    std::unordered_map<const Expr*, Tensor> partial_;
};

} // namespace

std::vector<Tensor> diff(const Tensor& expr, const std::vector<Tensor>& vars) {
    return Ad(expr).partial(vars);
}
