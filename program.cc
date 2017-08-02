#include "program.h"

#include <unordered_map>
#include <type_traits>

namespace {

template <class ReadRefResolver, class WriteRefResolver>
struct StatementExecutor {
    void operator()(const Tile& t) const {
        tile(x(), t.multiplier, result());
// FIXME
#if 0
        if (tile.multiplier.dim() != 2) {
            throw std::logic_error(
                    "Tile not implemented for shape " + tile.multiplier.toString());
        }
        result() = repmat(x(), tile.multiplier(0), tile.multiplier(1));
#endif
    }

    void operator()(const Untile& u) const {
        untile(x(), u.multiplier, result());
// FIXME
#if 0
        if (untile.multiplier.dim() != 2) {
            throw std::logic_error(
                    "Untile not implemented for shape " +
                    untile.multiplier.toString());
        }
        auto& r = result();
        const auto& tiled = x();
        r.fill(0.0f);
        for (size_t i = 0; i != untile.multiplier(0); ++i) {
            const size_t beginRow = i * untile.originalShape(0);
            for (size_t j = 0; j != untile.multiplier(1); ++j) {
                const size_t beginCol = j * untile.originalShape(1);
                r += tiled.submat(
                        beginRow,
                        beginCol,
                        // Subtract one because the ranges are inclusive here.
                        beginRow + untile.originalShape(0) - 1,
                        beginCol + untile.originalShape(1) - 1);
            }
        }
#endif
    }

    void operator()(const detail::FusedBinaryOp& binary) const {
        const auto& xMod = binary.xMod;
        const auto& yMod = binary.yMod;

        switch (binary.op) {
            case BinaryOperator::Plus:
                add(
                        x(),
                        xMod.transpose,
                        xMod.negate,
                        y(),
                        yMod.transpose,
                        yMod.negate,
                        result());
                break;
            case BinaryOperator::Mul:
                multiply(
                        x(),
                        xMod.transpose,
                        y(),
                        yMod.transpose,
                        xMod.negate ^ yMod.negate,
                        result());
                break;
            case BinaryOperator::HadamardMul:
                hadamard(
                        x(),
                        xMod.transpose,
                        y(),
                        yMod.transpose,
                        xMod.negate ^ yMod.negate,
                        result());
                break;
            case BinaryOperator::HadamardDiv:
                divide(
                        x(),
                        xMod.transpose,
                        y(),
                        yMod.transpose,
                        xMod.negate ^ yMod.negate,
                        result());
                break;
        }
    }

    void operator()(const Conv2D& conv) const {
        conv2d(x(), y(), conv, result());
// FIXME
#if 0
        const auto& a = x();
        const auto& k = y();

        const size_t kRows = k.n_rows;
        const size_t kCols = k.n_cols;

        for (size_t row = 0; row < result().n_rows; ++row) {
            for (size_t col = 0; col < result().n_cols; ++col) {
                int firstARow = (int)row - conv.padTop;
                int lastARow = firstARow + kRows;
                int firstACol = (int)col - conv.padLeft;
                int lastACol = firstACol + kCols;

                int firstKRow = 0;
                int lastKRow = kRows;
                int firstKCol = 0;
                int lastKCol = kCols;

                if (firstARow < 0) {
                    firstKRow = -firstARow;
                    firstARow = 0;
                }
                if (lastARow > (int)a.n_rows) {
                    lastKRow -= lastARow - a.n_rows;
                    lastARow = a.n_rows;
                }
                if (firstACol < 0) {
                    firstKCol = -firstACol;
                    firstACol = 0;
                }
                if (lastACol > (int)a.n_cols) {
                    lastKCol -= lastACol - a.n_cols;
                    lastACol = a.n_cols;
                }
                result()(row, col) = dot(
                        a.submat(
                            firstARow,
                            firstACol,
                            // The ranges are inclusive, so subtract one.
                            lastARow - 1,
                            lastACol - 1),
                        k.submat(
                            firstKRow,
                            firstKCol,
                            lastKRow - 1,
                            lastKCol - 1));
            }
        }
#endif
    }

    void operator()(const MaxPool& m) const {
        maxPool(x(), m.pool, result());
// FIXME
#if 0
        for (size_t row = 0; row != result().n_rows; ++row) {
            for (size_t col = 0; col != result().n_cols; ++col) {
                result()(row, col) = x().submat(
                        row * maxPool.rows,
                        col * maxPool.cols,
                        (row + 1) * maxPool.rows - 1,
                        (col + 1) * maxPool.cols - 1).max();
            }
        }
#endif
    }

    void operator()(const MaxPoolDiff& m) const {
        maxPoolDiff(x(), y(), z(), m.pool, result());
// FIXME
#if 0
        result().fill(0.0f);
        for (size_t row = 0; row != x().n_rows; ++row) {
            const size_t rowPool = row / maxPool.rows;
            for (size_t col = 0; col != x().n_cols; ++col) {
                const size_t colPool = col / maxPool.cols;
                if (x()(row, col) == y()(rowPool, colPool)) {
                    result()(row, col) = z()(rowPool, colPool);
                }
            }
        }
#endif
    }

    void operator()(const Pow& p) const {
        pow(x(), p.y, result());
    }

    void operator()(const Exp&) const {
        exp(x(), result());
    }

    void operator()(const Log&) const {
        log(x(), result());
    }

    void operator()(const Copy&) const {
        *result() = x();
    }

    void operator()(const Negate&) const {
        negate(x(), result());
    }

    void operator()(const Transpose&) const {
        transpose(x(), result());
    }

    void operator()(const Reverse&) const {
        reverse(x(), result());
// FIXME
#if 0
        result() = fliplr(flipud(x()));
#endif
    }

    void operator()(const Reshape&) const {
        reshape(x(), result());
// FIXME
#if 0
        result() = x();
        result().reshape(reshape.shape.rows, reshape.shape.cols);
#endif
    }

    void operator()(const Sigmoid&) const {
// FIXME
#if 0
        result() = 1.0f / (1.0f + exp(-x()));
#endif
    }

    void operator()(const HalfSumSquares&) const {
// FIXME
#if 0
        result() = accu(pow(x(), 2)) * 0.5f;
#endif
    }

    TensorValue* result() const { return writeRefResolver(stmt.result); }
    const TensorValue& x() const { return *readRefResolver(stmt.args.at(0)); }
    const TensorValue& y() const { return *readRefResolver(stmt.args.at(1)); }
    const TensorValue& z() const { return *readRefResolver(stmt.args.at(2)); }

    ReadRefResolver readRefResolver;
    WriteRefResolver writeRefResolver;
    const detail::Statement& stmt;
};

struct StatementFuser {
    detail::VmOp operator()(const BinaryOp& binaryOp) {
        return detail::FusedBinaryOp{
            binaryOp.op, fuse(args.at(0)), fuse(args.at(1))};
    }

    template <class T>
    detail::VmOp operator()(const T& t) {
        return t;
    }

    detail::VmOp operator()(const Const&) {
        throw std::logic_error("Unexpected op in StatementFuser");
    }

    detail::VmOp operator()(const Var&) {
        throw std::logic_error("Unexpected op in StatementFuser");
    }

    detail::VmOp operator()(const Placeholder&) {
        throw std::logic_error("Unexpected op in StatementFuser");
    }

    detail::NegateTranspose fuse(std::shared_ptr<Expr>& expr) {
        detail::NegateTranspose result;
        for (;;) {
            if (mpark::get_if<Transpose>(&expr->op)) {
                expr = expr->args.front();
                result.transpose = !result.transpose;
            } else if (mpark::get_if<Negate>(&expr->op)) {
                expr = expr->args.front();
                result.negate = !result.negate;
            } else {
                break;
            }
        }
        return result;
    }

    std::vector<std::shared_ptr<Expr>> args;
};

class Compiler {
public:
    Compiler(
            std::vector<Tensor> targets,
            std::vector<Tensor> args) :
        targets_(std::move(targets))
    {
        for (size_t i = 0; i != args.size(); ++i) {
            if (const auto ph =
                    mpark::get_if<Placeholder>(&args[i].unwrap()->op)) {
                argToIndex_[ph] = i;
            } else {
                throw std::logic_error(
                        "All arguments in compile() must be placeholders");
            }
        }
        for (size_t i = 0; i != targets_.size(); ++i) {
            const auto& target = targets_[i];
            exprToResultIndex_[target.unwrap().get()] = i;
            const auto shape = target.shape();
            // Preallocate result.
            result_.push_back(TensorValue::zeros(shape));
        }
    }

    // TODO Detect common sub-expressions
    Program compile() && {
        for (const auto& target: targets_) {
            compile(target.unwrap());
        }
        return Program(
                std::move(program_),
                std::move(tmp_),
                std::move(result_),
                std::move(retainer_));
    }

private:
    // In the special case when function result is a value of consant,
    // variable or function argument, we need co explicitly copy tensor
    // to result location.
    detail::ReadRef addCopyStatement(
            const Expr* expr, const detail::ReadRef& ref) {
        auto iter = exprToResultIndex_.find(expr);
        if (iter == exprToResultIndex_.end()) {
            return ref;
        }
        program_.push_back({
            detail::VmOp{Copy{}},
            std::vector<detail::ReadRef>{ref},
            detail::WriteRef{detail::ResultRef{iter->second}}});
        return ref;
    }


    detail::ReadRef compile(const std::shared_ptr<Expr>& expr) {
        auto iter = compiled_.find(expr.get());
        if (iter != compiled_.end()) {
            return iter->second;
        }

        auto ref = doCompile(expr);
        compiled_.emplace(expr.get(), ref);
        return ref;
    }

    detail::ReadRef doCompile(const std::shared_ptr<Expr>& expr) {
        // Deal with variable refs.
        if (const auto var = mpark::get_if<Var>(&expr->op)) {
            retainer_.push_back(expr);
            return addCopyStatement(expr.get(), detail::VarRef{&var->value});
        } else if (const auto ph = mpark::get_if<Placeholder>(&expr->op)) {
            auto iter = argToIndex_.find(ph);
            if (iter == argToIndex_.end()) {
                throw std::runtime_error("Unbound variable");
            }
            return addCopyStatement(expr.get(), detail::ArgRef{iter->second});
        } else if (const auto konst = mpark::get_if<Const>(&expr->op)) {
            retainer_.push_back(expr);
            return addCopyStatement(expr.get(), detail::ConstRef{&konst->value});
        }

        const auto shape = expr->shape;
        auto iter = exprToResultIndex_.find(expr.get());
        const auto ref = [&]() -> detail::WriteRef {
            if (iter == exprToResultIndex_.end()) {
                // Allocate new tensor.
                tmp_.push_back(TensorValue::zeros(shape));

                return detail::TmpRef{tmp_.size() - 1};
            } else {
                return detail::ResultRef{iter->second};
            }
        }();

        fuseStatement(expr.get(), ref);

        // Cast write ref to read ref.
        return mpark::visit(
                [](auto ref) -> detail::ReadRef { return ref; }, ref);
    }

    void fuseStatement(const Expr* expr, const detail::WriteRef& ref) {
        StatementFuser fuser{expr->args};
        const auto op = mpark::visit(fuser, expr->op);

        std::vector<detail::ReadRef> args;
        for (const auto& arg: fuser.args) {
            args.push_back(compile(arg));
        }
        program_.push_back({op, std::move(args), ref});
    }

    std::vector<Tensor> targets_;
    std::unordered_map<const Expr*, size_t> exprToResultIndex_;
    std::unordered_map<const Placeholder*, size_t> argToIndex_;

    std::unordered_map<const Expr*, detail::ReadRef> compiled_;
    std::vector<detail::Statement> program_;
    std::vector<TensorValue> tmp_;
    std::vector<TensorValue> result_;
    std::vector<std::shared_ptr<Expr>> retainer_;
};

template <class Result>
struct RefResolver {
    Result operator()(const detail::ArgRef& ref) const {
        return args.at(ref.index);
    }

    Result operator()(const detail::ResultRef& ref) const {
        return &result.at(ref.index);
    }

    Result operator()(const detail::TmpRef& ref) const {
        return &tmp.at(ref.index);
    }

    Result operator()(const detail::VarRef& ref) const {
        return ref.value;
    }

    Result operator()(const detail::ConstRef& ref) const {
        return ref.value;
    }

    const std::vector<const TensorValue *>& args;
    std::vector<TensorValue>& result;
    std::vector<TensorValue>& tmp;
};

struct PrettyPrinter {
    void operator()(const detail::ResultRef& ref) const {
        out << "result[" << ref.index << "]";
    }

    void operator()(const detail::TmpRef& ref) const {
        out << "tmp[" << ref.index << "]";
    }

    void operator()(const detail::ArgRef& ref) const {
        out << "arg[" << ref.index << "]";
    }

    void operator()(const detail::VarRef& ref) const {
        out << "var(" << ref.value << ")";
    }

    void operator()(const detail::ConstRef& ref) const {
        if (ref.value->shape().isScalar()) {
            out << "const(" << ref.value->asScalar() << ")";
        } else {
            out << "const(" << ref.value << ")";
        }
    }

    void operator()(const Tile& tile) const {
        out << "tile<" << tile.multiplier.toString() << ">";
    }

    void operator()(const Untile& untile) const {
        out << "untile<" << untile.multiplier.toString() << ">";
    }

    void operator()(const detail::FusedBinaryOp& binary) const {
        switch (binary.op) {
            case BinaryOperator::Plus:
                out << "+";
                break;
            case BinaryOperator::Mul:
                out << "*";
                break;
            case BinaryOperator::HadamardMul:
                out << "%";
                break;
            case BinaryOperator::HadamardDiv:
                out << "/";
                break;
        }
        out << "<x:";
        if (binary.xMod.negate) {
            out << " n";
        }
        if (binary.xMod.transpose) {
            out << " t";
        }
        out << "; y:";
        if (binary.yMod.negate) {
            out << " n";
        }
        if (binary.yMod.transpose) {
            out << " t";
        }
        out << ">";
    }

    void operator()(const Conv2D&) const {
        out << "conv2";
    }

    void operator()(const MaxPool& maxPool) const {
        out << "maxPool<" << maxPool.pool.toString() << ">";
    }

    void operator()(const MaxPoolDiff& maxPool) const {
        out << "maxPoolDiff<" << maxPool.pool.toString() << ">";
    }

    void operator()(const Pow& pow) {
        out << "pow<" << pow.y << ">";
    }

    void operator()(const Exp&) {
        out << "exp";
    }

    void operator()(const Log&) {
        out << "log";
    }

    void operator()(const Copy&) {
        out << "copy";
    }

    void operator()(const Negate&) {
        out << "negate";
    }

    void operator()(const Transpose&) {
        out << "transpose";
    }

    void operator()(const Reverse&) {
        out << "reverse";
    }

    void operator()(const Reshape&) {
        out << "reshape";
    }

    void operator()(const Sigmoid&) {
        out << "sigmoid";
    }

    void operator()(const HalfSumSquares&) {
        out << "halfSumSquares";
    }

    std::ostream& out;
};

} // namespace


Program compile(
        const std::vector<Tensor> &targets,
        const std::vector<Tensor>& args) {
    return Compiler(targets, args).compile();
}

const std::vector<TensorValue>& Program::operator()(
        const std::vector<const TensorValue*>& args) {
    auto resolveRead = [&](const detail::ReadRef& ref) {
        return mpark::visit(
                RefResolver<const TensorValue *>{args, result_, tmp_},
                ref);
    };
    auto resolveWrite = [&](const detail::WriteRef& ref) {
        return mpark::visit(
                RefResolver<TensorValue *>{args, result_, tmp_},
                ref);
    };
    for (auto& statement: program_) {
        mpark::visit(
                StatementExecutor<
                    decltype(resolveRead),
                    decltype(resolveWrite)>
                {resolveRead, resolveWrite, statement}, statement.op);
    }
    return result_;
}

std::ostream& operator <<(std::ostream& out, const Program& program) {
    PrettyPrinter pp{out};
    for (const auto& s: program.program_) {
        mpark::visit(pp, s.result);
        out << " = ";
        mpark::visit(pp, s.op);
        out << "(";
        for (size_t i = 0; i != s.args.size(); ++i) {
            if (i != 0) {
                out << ", ";
            }
            mpark::visit(pp, s.args[i]);
        }
        out << ");\n";
    }
    return out;
}
