#include "program.h"

#include <unordered_map>

namespace {

template <class ReadRefResolver, class WriteRefResolver>
struct StatementExecutor {
    void operator()(const Tile& tile) const {
        result() = repmat(x(), tile.repeatRows, tile.repeatCols);
    }

    void operator()(const Untile& untile) const {
        auto& r = result();
        const auto& tiled = x();
        r.fill(0.0f);
        for (size_t i = 0; i != untile.repeatRows; ++i) {
            const size_t beginRow = i * untile.originalShape.rows;
            for (size_t j = 0; j != untile.repeatCols; ++j) {
                const size_t beginCol = j * untile.originalShape.cols;
                r += tiled.submat(
                        beginRow,
                        beginCol,
                        // Subtract one because the ranges are inclusive here.
                        beginRow + untile.originalShape.rows - 1,
                        beginCol + untile.originalShape.cols - 1);
            }
        }
    }

    void operator()(const BinaryOp& binary) const {
        switch (binary.op) {
            case BinaryOperator::Plus:
                result() = x() + y();
                break;
            case BinaryOperator::Minus:
                result() = x() - y();
                break;
            case BinaryOperator::Mul:
                result() = x() * y();
                break;
            case BinaryOperator::HadamardMul:
                result() = x() % y();
                break;
            case BinaryOperator::HadamardDiv:
                result() = x() / y();
                break;
        }
    }

    void operator()(const Pow& pow) const {
        result() = arma::pow(x(), pow.y);
    }

    void operator()(const Exp&) const {
        result() = arma::exp(x());
    }

    void operator()(const Log&) const {
        result() = arma::log(x());
    }

    void operator()(const Copy&) const {
        result() = x();
    }

    void operator()(const Negate&) const {
        result() = -x();
    }

    void operator()(const Transpose&) const {
        result() = x().t();
    }

    void operator()(const Reshape& reshape) const {
        result() = x();
        result().reshape(reshape.shape.rows, reshape.shape.cols);
    }

    void operator()(const Sigmoid&) const {
        result() = 1.0f / (1.0f + exp(-x()));
    }

    template <class T>
    void operator()(const T&) const {
        throw std::logic_error("Unexpected op in StatementExecutor");
    }

    Matrix& result() const { return *writeRefResolver(stmt.result); }
    const Matrix& x() const { return *readRefResolver(stmt.args.at(0)); }
    const Matrix& y() const { return *readRefResolver(stmt.args.at(1)); }

    ReadRefResolver readRefResolver;
    WriteRefResolver writeRefResolver;
    detail::Statement stmt;
};

class Compiler {
public:
    Compiler(
            std::vector<Tensor> targets,
            std::vector<std::string> args) :
        targets_(std::move(targets)),
        args_(std::move(args))
    {
        for (size_t i = 0; i != args_.size(); ++i) {
            argNameToIndex_[args_[i]] = i;
        }
        result_.resize(targets_.size());
        for (size_t i = 0; i != targets_.size(); ++i) {
            const auto& target = targets_[i];
            exprToResultIndex_[target.unwrap().get()] = i;
            const auto shape = target.shape();
            // Preallocate result.
            result_[i] = Matrix(shape.rows, shape.cols);
        }
    }

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
    // variable or function argument, we need co explicitly copy matrix
    // to result location.
    detail::ReadRef addCopyStatement(
            const Expr* expr, const detail::ReadRef& ref) {
        auto iter = exprToResultIndex_.find(expr);
        if (iter == exprToResultIndex_.end()) {
            return ref;
        }
        program_.push_back({
            Op{Copy{}},
            std::vector<detail::ReadRef>{ref},
            detail::WriteRef{detail::ResultRef{iter->second}}});
        return ref;
    }


    detail::ReadRef compile(const std::shared_ptr<Expr>& expr) {
        // If already compiled return
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
            if (const auto name = mpark::get_if<std::string>(&var->state)) {
                auto iter = argNameToIndex_.find(*name);
                if (iter == argNameToIndex_.end()) {
                    throw std::runtime_error("Unbound variable :" + *name);
                }
                return addCopyStatement(expr.get(), detail::ArgRef{iter->second});
            } else {
                const auto& matrix = mpark::get<Matrix>(var->state);
                retainer_.push_back(expr);
                return addCopyStatement(expr.get(), detail::VarRef{&matrix});
            }
        } else if (const auto konst = mpark::get_if<Const>(&expr->op)) {
            retainer_.push_back(expr);
            return addCopyStatement(expr.get(), detail::VarRef{&konst->value});
        }

        std::vector<detail::ReadRef> args;
        for (const auto& arg: expr->args) {
            args.push_back(compile(arg));
        }

        const auto shape = expr->shape;
        auto iter = exprToResultIndex_.find(expr.get());
        const auto ref = [&]() -> detail::WriteRef {
            if (iter == exprToResultIndex_.end()) {
                // Allocate new matrix.
                tmp_.emplace_back(shape.rows, shape.cols);

                return detail::TmpRef{tmp_.size() - 1};
            } else {
                return detail::ResultRef{iter->second};
            }
        }();
        program_.push_back({expr->op, std::move(args), ref});
        // Cast write ref to read ref.
        return mpark::visit(
                [](auto ref) -> detail::ReadRef { return ref; }, ref);
    }

    std::vector<Tensor> targets_;
    std::unordered_map<const Expr*, size_t> exprToResultIndex_;
    std::vector<std::string> args_;
    std::unordered_map<std::string, size_t> argNameToIndex_;

    std::unordered_map<const Expr*, detail::ReadRef> compiled_;
    std::vector<detail::Statement> program_;
    std::vector<Matrix> tmp_;
    std::vector<Matrix> result_;
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
        return ref.matrix;
    }

    const std::vector<const Matrix *>& args;
    std::vector<Matrix>& result;
    std::vector<Matrix>& tmp;
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
        out << "var(" << ref.matrix << ")";
    }

    void operator()(const Tile& tile) const {
        out << "tile<" << tile.repeatRows << ", " << tile.repeatCols << ">";
    }

    void operator()(const Untile& untile) const {
        out << "untile<" << untile.repeatRows << ", " << untile.repeatCols << ">";
    }

    void operator()(const BinaryOp& op) const {
        switch (op.op) {
            case BinaryOperator::Plus:
                out << "+";
                break;
            case BinaryOperator::Minus:
                out << "-";
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

    void operator()(const Reshape& reshape) {
        out << "reshape<" << reshape.shape.rows << ", " <<
            reshape.shape.cols << ">";
    }

    void operator()(const Sigmoid&) {
        out << "sigmoid";
    }

    template <class T>
    void operator()(const T&) const {
        throw std::logic_error("Unexpected op in PrettyPrinter");
    }

    std::ostream& out;
};

} // namespace


// TODO implement operations fusion
Program compile(
        const std::vector<Tensor> &targets,
        const std::vector<std::string>& args) {
    return Compiler(targets, args).compile();
}

void Program::execute(
        const detail::Statement& stmt,
        const std::vector<const Matrix *>& args) {
    auto resolveRead = [&](const detail::ReadRef& ref) {
        return mpark::visit(
                RefResolver<const Matrix *>{args, result_, tmp_},
                ref);
    };
    auto resolveWrite = [&](const detail::WriteRef& ref) {
        return mpark::visit(
                RefResolver<Matrix *>{args, result_, tmp_},
                ref);
    };
    mpark::visit(
            StatementExecutor<
                decltype(resolveRead),
                decltype(resolveWrite)>
            {resolveRead, resolveWrite, stmt}, stmt.op);
}

const std::vector<Matrix>& Program::operator()(
        const std::vector<const Matrix*>& args) {
    for (auto& s: program_) {
        execute(s, args);
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
