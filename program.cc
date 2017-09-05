#include "program.h"

#include <iostream>
#include <type_traits>
#include <unordered_map>

namespace {

struct Reshaper {
    detail::ReadRef operator()(const detail::ArgRef& ref) {
        return detail::ArgRef{ref.index, std::move(shape)};
    }

    detail::ReadRef operator()(const ConstTensorRef& ref) {
        return ref.reshape(std::move(shape));
    }

    Shape shape;
};

struct StatementExecutor {
    void operator()(const Tile& t) {
        tile(x(), t.multiplier, result());
    }

    void operator()(const Untile& u) {
        untile(x(), u.multiplier, result());
    }

    void operator()(const detail::FusedBinaryOp& binary) {
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

    void operator()(const Conv2D& conv) {
        conv2d(x(), y(), conv, result());
    }

    void operator()(const MaxPool2D& m) {
        maxPool2d(x(), m.rows, m.cols, result());
    }

    void operator()(const MaxPool2DDiff& m) {
        maxPoolDiff2d(x(), y(), z(), m.rows, m.cols, result());
    }

    void operator()(const Pow& p) {
        pow(x(), p.y, result());
    }

    void operator()(const Exp&) {
        exp(x(), result());
    }

    void operator()(const Log&) {
        log(x(), result());
    }

    void operator()(const Copy&) {
        std::copy(x().data(), x().dataEnd(), result().data());
    }

    void operator()(const Negate&) {
        negate(x(), result());
    }

    void operator()(const Transpose&) {
        transpose(x(), result());
    }

    void operator()(const Reverse&) {
        reverse(x(), result());
    }

    void operator()(const Sigmoid&) {
        sigmoid(x(), result());
    }

    void operator()(const HalfSumSquares&) {
        halfSumSquares(x(), result());
    }

    TensorRef&& result() { return std::move(stmt.result); }
    ConstTensorRef x() { return argAt(0); }
    ConstTensorRef y() { return argAt(1); }
    ConstTensorRef z() { return argAt(2); }

    ConstTensorRef argAt(size_t index) {
        return mpark::visit(*this, stmt.args.at(index));
    }

    ConstTensorRef operator()(const detail::ArgRef& ref) {
        return ConstTensorRef{
            ref.shape,
            args.at(ref.index)->data()};
    }

    ConstTensorRef operator()(const ConstTensorRef& ref) {
        return ref;
    }

    detail::Statement& stmt;
    const std::vector<const TensorValue *>& args;
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

    detail::VmOp operator()(const Reshape&) {
        throw std::logic_error("Unexpected op in StatementFuser");
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
            const auto* target = targets_[i].unwrap().get();
            // Allocate result with correct shape
            result_.push_back(TensorValue::zeros(target->shape));
            // Then skip all the reshapes and consider that expression
            // an actual target computation that would write to the result.
            while (mpark::get_if<Reshape>(&target->op)) {
                target = target->args.at(0).get();
            }
            if (exprToResultIndex_.count(target)) {
                throw std::logic_error(
                        "Compiler: support for aliased results is not implemented");
            }
            exprToResultIndex_[target] = i;
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
            TensorRef{result_.at(iter->second)}});
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
            return addCopyStatement(expr.get(), ConstTensorRef{var->value});
        } else if (const auto ph = mpark::get_if<Placeholder>(&expr->op)) {
            auto iter = argToIndex_.find(ph);
            if (iter == argToIndex_.end()) {
                throw std::runtime_error("Unbound variable");
            }
            return addCopyStatement(
                    expr.get(), detail::ArgRef{iter->second, expr->shape});
        } else if (const auto konst = mpark::get_if<Const>(&expr->op)) {
            retainer_.push_back(expr);
            return addCopyStatement(expr.get(), ConstTensorRef{&konst->value});
        }

        if (const auto reshape = mpark::get_if<Reshape>(&expr->op)) {
            return mpark::visit(Reshaper{expr->shape}, compile(expr->args.at(0)));
        }

        const auto shape = expr->shape;
        auto iter = exprToResultIndex_.find(expr.get());
        const auto ref = [&]() -> TensorRef {
            if (iter == exprToResultIndex_.end()) {
                // Allocate new tensor.
                tmp_.push_back(TensorValue::zeros(shape));

                return tmp_.back();
            } else {
                return TensorRef{expr->shape, result_.at(iter->second).data()};
            }
        }();

        fuseStatement(expr.get(), ref);

        return ConstTensorRef{ref};
    }

    void fuseStatement(const Expr* expr, TensorRef ref) {
        StatementFuser fuser{expr->args};
        const auto op = mpark::visit(fuser, expr->op);

        std::vector<detail::ReadRef> args;
        for (const auto& arg: fuser.args) {
            args.push_back(compile(arg));
        }
        program_.push_back({op, std::move(args), std::move(ref)});
    }

    std::vector<Tensor> targets_;
    std::unordered_map<const Expr*, size_t> exprToResultIndex_;
    std::unordered_map<const Placeholder*, size_t> argToIndex_;

    std::unordered_map<const Expr*, detail::ReadRef> compiled_;
    std::vector<detail::Statement> program_;
    std::deque<TensorValue> tmp_;
    std::vector<TensorValue> result_;
    std::vector<std::shared_ptr<Expr>> retainer_;
};

template <class Result>
struct RefResolver {
};

struct PrettyPrinter {
    void operator()(const ConstTensorRef& ref) {
        out << "cref(" << varName(ref.data()) << ")";
    }

    void operator()(const TensorRef& ref) {
        out << "ref(" << varName(ref.data()) << ")";
    }

    void operator()(const detail::ArgRef& ref) {
        out << "arg[" << ref.index << "]";
    }

    void operator()(const Tile& tile) {
        out << "tile<" << tile.multiplier.toString() << ">";
    }

    void operator()(const Untile& untile) {
        out << "untile<" << untile.multiplier.toString() << ">";
    }

    void operator()(const detail::FusedBinaryOp& binary) {
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

    void operator()(const Conv2D&) {
        out << "conv2";
    }

    void operator()(const MaxPool2D& maxPool) {
        out << "maxPool2d<" << maxPool.rows << ", " << maxPool.cols << ">";
    }

    void operator()(const MaxPool2DDiff& maxPool) {
        out << "maxPoolDiff2d<" << maxPool.rows << ", " << maxPool.cols << ">";
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

    void operator()(const Sigmoid&) {
        out << "sigmoid";
    }

    void operator()(const HalfSumSquares&) {
        out << "halfSumSquares";
    }

    std::string varName(const float* addr) {
        auto iter = refRegistry.find(addr);
        if (iter == refRegistry.end()) {
            iter = refRegistry.emplace(addr, refRegistry.size()).first;
        }
        return "x" + std::to_string(iter->second);
    }

    std::ostream& out;
    std::unordered_map<const float *, size_t> refRegistry;
};

} // namespace


Program compile(
        const std::vector<Tensor> &targets,
        const std::vector<Tensor>& args) {
    return Compiler(targets, args).compile();
}

const std::vector<TensorValue>& Program::operator()(
        const std::vector<const TensorValue*>& args) {
    for (auto& statement: program_) {
        mpark::visit(StatementExecutor{statement, args}, statement.op);
    }
    return result_;
}

std::ostream& operator <<(std::ostream& out, const Program& program) {
    PrettyPrinter pp{out, {}};
    for (const auto& s: program.program_) {
        pp(s.result);
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
