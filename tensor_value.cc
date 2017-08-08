#include "tensor_value.h"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <random>
#include <stdexcept>
#include <iostream>

#include <cblas.h>

namespace {

template <class X, class Y, class Op>
void zipWithImpl(X x, Y y, Op op, TensorRef&& result) {
    const size_t size = result.shape().size();
    for (size_t i = 0; i != size; ++i) {
        result.data()[i] = op(x(i), y(i));
    }
}

template <class Op>
void zipWith(
        const ConstTensorRef& x,
        const ConstTensorRef& y,
        Op op,
        TensorRef&& result) {
    auto xi = [&](size_t i) { return x.data()[i]; };
    auto yi = [&](size_t i) { return y.data()[i]; };
    auto cont = [&](auto xi, auto yi) { zipWithImpl(xi, yi, op, std::move(result)); };
    if (x.shape().isScalar()) {
        cont([value = x.toScalar()](size_t) { return value; }, yi);
    } else if (y.shape().isScalar()) {
        cont(xi, [value = y.toScalar()](size_t) { return value; });
    } else {
        cont(xi, yi);
    }
}

template <class Op>
void map(const ConstTensorRef& x, Op op, TensorRef&& result) {
    const size_t size = x.shape().size();

    for (size_t i = 0; i != size; ++i) {
        result.data()[i] = op(x.data()[i]);
    }
}

float dot(
        const float* a,
        size_t aStride,
        const float* b,
        size_t bStride,
        size_t rows,
        size_t cols) {
    float result = 0;
    for (size_t i = 0; i != rows; ++i) {
        for (size_t j = 0; j != cols; ++j) {
            result += a[i * aStride + j] * b[i * bStride + j];
        }
    }
    return result;
}

std::default_random_engine& randomEngine() {
    static std::default_random_engine generator;
    return generator;
}

template <class Dist>
TensorValue randomTensor(const Shape& shape, Dist&& dist) {
    std::vector<float> data(shape.size());
    for (float& x: data) {
        x = dist(randomEngine());
    }
    return TensorValue{shape, std::move(data)};
}

} // namespace

namespace detail {

template <class T>
ConstTensorBase<T>::ConstTensorBase(Shape shape) :
    shape_(std::move(shape))
{}

template <class T>
const float& ConstTensorBase<T>::operator()(size_t i) const {
    if (shape_.dim() != 1) {
        throw std::logic_error("Not a vector, shape = " + shape_.toString());
    }
    return getData()[i];
}

template <class T>
const float& ConstTensorBase<T>::operator()(size_t i, size_t j) const {
    if (shape_.dim() != 2) {
        throw std::logic_error("Not a matrix, shape = " + shape_.toString());
    }
    return getData()[i * shape_(1) + j];
}

template <class T>
const float& ConstTensorBase<T>::operator()(size_t i, size_t j, size_t k) const {
    if (shape_.dim() != 3) {
        throw std::logic_error("Not a cube, shape = " + shape_.toString());
    }
    return getData()[(i * shape_(1) + j) * shape_(2) + k];
}

template <class T>
const float& ConstTensorBase<T>::operator()(const std::vector<size_t> &indices) const {
    if (shape_.dim() != indices.size()) {
        throw std::logic_error(
                "Unexpected number of indices: " + std::to_string(indices.size()) +
                " for shape " + shape_.toString() );
    }

    size_t index = 0;
    for (size_t i = 0; i != indices.size(); ++i) {
        index = index * shape_(i) + indices[i];
    }

    return getData()[index];
}

template <class T>
float ConstTensorBase<T>::toScalar() const {
    if (shape_.size() != 1) {
        throw std::logic_error(
                "TensorValue::toScalar: cannot be converted to scalar");
    }
    return *getData();
}

template <class T>
TensorBase<T>::TensorBase(Shape shape) :
    ConstTensorBase<T>(std::move(shape))
{}

template <class T>
const float* TensorBase<T>::dataEnd() const {
    return static_cast<const ConstTensorBase<T> *>(this)->dataEnd();
}

template <class T>
const float& TensorBase<T>::operator()(size_t i) const {
    return static_cast<const ConstTensorBase<T> &>(*this)(i);
}

template <class T>
const float& TensorBase<T>::operator()(size_t i, size_t j) const {
    return static_cast<const ConstTensorBase<T> &>(*this)(i, j);
}

template <class T>
const float& TensorBase<T>::operator()(size_t i, size_t j, size_t k) const {
    return static_cast<const ConstTensorBase<T> &>(*this)(i, j, k);
}

template <class T>
const float& TensorBase<T>::operator()(const std::vector<size_t> &indices) const {
    return static_cast<const ConstTensorBase<T> &>(*this)(indices);
}

template <class T>
float& TensorBase<T>::operator()(size_t i) {
    return const_cast<float &>(const_cast<const TensorBase<T> &>(*this)(i));
}

template <class T>
float& TensorBase<T>::operator()(size_t i, size_t j) {
    return const_cast<float &>(const_cast<const TensorBase<T> &>(*this)(i, j));
}

template <class T>
float& TensorBase<T>::operator()(size_t i, size_t j, size_t k) {
    return const_cast<float &>(const_cast<const TensorBase<T> &>(*this)(i, j, k));
}

template <class T>
float& TensorBase<T>::operator()(const std::vector<size_t> &indices) {
    return const_cast<float &>(const_cast<const TensorBase<T> &>(*this)(indices));
}

} // namespace detail

TensorValue::TensorValue(float x) :
    TensorBase(Shape{}), data_(1, x)
{}

TensorValue::TensorValue(Shape shape, std::vector<float> data) :
    detail::TensorBase<TensorValue>(std::move(shape)), data_(std::move(data))
{
    assert(!data_.empty());
    assert(data_.size() == this->shape().size());
}

TensorValue TensorValue::zeros(const Shape& shape) {
    return TensorValue{
        shape, std::vector<float>(shape.size(), 0.0f)};
}

TensorValue TensorValue::ones(const Shape& shape) {
    return TensorValue{
        shape, std::vector<float>(shape.size(), 1.0f)};
}

TensorValue TensorValue::randn(const Shape& shape, float stddev) {
    return randomTensor(shape, std::normal_distribution<float>(0, stddev));
}

TensorValue TensorValue::randu(const Shape& shape) {
    return randomTensor(shape, std::uniform_real_distribution<float>(0, 1));
}

ConstTensorRef::ConstTensorRef(const TensorValue& value) :
    detail::ConstTensorBase<ConstTensorRef>(value.shape()), data_(value.data())
{}

ConstTensorRef::ConstTensorRef(const TensorRef& ref) :
    detail::ConstTensorBase<ConstTensorRef>(ref.shape()), data_(ref.data())
{}

TensorRef::TensorRef(TensorValue& value) :
    TensorRef(&value)
{}

TensorRef::TensorRef(TensorValue* value) :
    detail::TensorBase<TensorRef>(value->shape()), data_(value->data())
{}

template class detail::ConstTensorBase<TensorValue>;
template class detail::TensorBase<TensorValue>;
template class detail::ConstTensorBase<ConstTensorRef>;
template class detail::TensorBase<TensorRef>;

void multiply(
        const ConstTensorRef& x,
        bool transposeX,
        const ConstTensorRef& y,
        bool transposeY,
        bool negateResult,
        TensorRef&& result) {
    if (x.shape().dim() != 2 || y.shape().dim() != 2) {
        throw std::logic_error(
                "Cannot multiply tensors with shape " + x.shape().toString() +
                " and " + y.shape().toString());
    }
    const float alpha = negateResult ? -1.0f : 1.0;

    const size_t m = x.shape()(transposeX);
    const size_t n = y.shape()(!transposeY);
    const size_t k = x.shape()(!transposeX);

    cblas_sgemm(
            CblasRowMajor,
            transposeX ? CblasTrans : CblasNoTrans,
            transposeY ? CblasTrans : CblasNoTrans,
            m,
            n,
            k,
            alpha,
            x.data(),
            x.shape()(1),
            y.data(),
            y.shape()(1),
            0.0f,
            result.data(),
            result.shape()(1));
}

void add(
        const ConstTensorRef& x,
        bool transposeX,
        bool negateX,
        const ConstTensorRef& y,
        bool transposeY,
        bool negateY,
        TensorRef&& result)
{
    assert(!transposeX && !transposeY);
    (void)transposeX;
    (void)transposeY;
    auto cont = [&](auto op) {
        zipWith(x, y, op, std::move(result));
    };
    if (negateX && negateY) {
        cont([](float x, float y) { return -x - y; });
    } else if (negateX && !negateY) {
        cont([](float x, float y) { return y - x; });
    } else if (!negateX && negateY) {
        cont([](float x, float y) { return x - y; });
    } else {
        cont([](float x, float y) { return x + y; });
    }
}

void divide(
        const ConstTensorRef& x,
        bool transposeX,
        const ConstTensorRef& y,
        bool transposeY,
        bool negateResult,
        TensorRef&& result)
{
    assert(!transposeX && !transposeY);
    (void)transposeX;
    (void)transposeY;
    auto cont = [&](auto op) {
        zipWith(x, y, op, std::move(result));
    };
    if (negateResult) {
        cont([](float x, float y) { return -x / y; });
    } else {
        cont([](float x, float y) { return x / y; });
    }
}

void hadamard(
        const ConstTensorRef& x,
        bool transposeX,
        const ConstTensorRef& y,
        bool transposeY,
        bool negateResult,
        TensorRef&& result)
{
    assert(!transposeX && !transposeY);
    (void)transposeX;
    (void)transposeY;
    auto cont = [&](auto op) {
        zipWith(x, y, op, std::move(result));
    };
    if (negateResult) {
        cont([](float x, float y) { return -x * y; });
    } else {
        cont([](float x, float y) { return x * y; });
    }
}

void pow(const ConstTensorRef& x, float y, TensorRef&& result) {
    map(x, [y](float x) { return std::pow(x, y); }, std::move(result));
}

void exp(const ConstTensorRef& x, TensorRef&& result) {
    map(x, [](float x) { return std::exp(x); }, std::move(result));
}

void log(const ConstTensorRef& x, TensorRef&& result) {
    map(x, [](float x) { return std::log(x); }, std::move(result));
}

void negate(const ConstTensorRef& x, TensorRef&& result) {
    map(x, [](float x) { return -x; }, std::move(result));
}

void transpose(const ConstTensorRef& x, TensorRef&& result) {
    const size_t rows = x.shape()(x.shape().dim() - 2);
    const size_t cols = x.shape()(x.shape().dim() - 1);
    for (size_t i = 0; i != rows; ++i) {
        for (size_t j = 0; j != cols; ++j) {
            result(j, i) = x(i, j);
        }
    }
}

void reverse(const ConstTensorRef& x, TensorRef&& result) {
    const size_t size = x.shape().size();
    for (size_t i = 0; i != size; ++i) {
        result.data()[size - i - 1] = x.data()[i];
    }
}

void reshape(const ConstTensorRef& x, TensorRef&& result) {
    std::copy(x.data(), x.dataEnd(), result.data());
}

float accu(const ConstTensorRef& x) {
    return std::accumulate(x.data(), x.dataEnd(), 0.0f);
}

void addMultiply(
        const ConstTensorRef& x, float factor, TensorRef&& y) {
    zipWith(
            x,
            y,
            [factor](float x, float y) { return x * factor + y; },
            std::move(y));
}

void tile(const ConstTensorRef& x, const Shape& multiplier, TensorRef&& y) {
    std::vector<size_t> indices(multiplier.dim(), 0);
    std::vector<size_t> xIndices(multiplier.dim());
    const size_t size = y.shape().size();
    for (size_t i = 0; i != size; ++i) {
        for (size_t j = 0; j != xIndices.size(); ++j) {
            xIndices[j] = indices[j] % x.shape()(j);
        }
        y(indices) = x(xIndices);
        if (i + 1 != size) {
            size_t j = 0;
            for (; indices[j] == y.shape()(j) - 1; ++j) {
                indices[j] = 0;
            }
            ++indices[j];
        }
    }
}

void untile(const ConstTensorRef& x, const Shape& multiplier, TensorRef&& y) {
    std::vector<size_t> indices(multiplier.dim(), 0);
    std::vector<size_t> yIndices(multiplier.dim());
    const size_t size = x.shape().size();
    std::fill(y.data(), y.dataEnd(), 0.0f);
    for (size_t i = 0; i != size; ++i) {
        for (size_t j = 0; j != yIndices.size(); ++j) {
            yIndices[j] = indices[j] % y.shape()(j);
        }
        y(yIndices) += x(indices);
        if (i + 1 != size) {
            size_t j = 0;
            for (; indices[j] == x.shape()(j) - 1; ++j) {
                indices[j] = 0;
            }
            ++indices[j];
        }
    }
}

void sigmoid(const ConstTensorRef& x, TensorRef&& result) {
    map(x, [](float x) { return 1.0f / (1.0f + std::exp(-x)); }, std::move(result));
}

void halfSumSquares(const ConstTensorRef& x, TensorRef&& result) {
    *result.data() = 0.5f * std::accumulate(
            x.data(),
            x.dataEnd(),
            0.0f,
            [](float x, float y) { return x + y * y; });
}

void conv2d(
        const ConstTensorRef& a,
        const ConstTensorRef& k,
        const Conv2D& conv,
        TensorRef&& result) {
    const size_t kRows = k.shape()(0);
    const size_t kCols = k.shape()(1);

    for (size_t row = 0; row < result.shape()(0); ++row) {
        for (size_t col = 0; col < result.shape()(1); ++col) {
            int firstARow = (int)row - conv.padTop;
            int lastARow = firstARow + kRows;
            int firstACol = (int)col - conv.padLeft;
            int lastACol = firstACol + kCols;

            int firstKRow = 0;
            int firstKCol = 0;

            if (firstARow < 0) {
                firstKRow = -firstARow;
                firstARow = 0;
            }
            if (lastARow > (int)a.shape()(0)) {
                lastARow = a.shape()(0);
            }
            if (firstACol < 0) {
                firstKCol = -firstACol;
                firstACol = 0;
            }
            if (lastACol > (int)a.shape()(1)) {
                lastACol = a.shape()(1);
            }
            result(row, col) = dot(
                    &a(firstARow, firstACol),
                    a.shape()(0),
                    &k(firstKRow, firstKCol),
                    k.shape()(0),
                    lastARow - firstARow,
                    lastACol - firstACol);
        }
    }
}

void maxPool(
        const ConstTensorRef& a, const Shape& pool, TensorRef&& result) {
    for (size_t row = 0; row != result.shape()(0); ++row) {
        for (size_t col = 0; col != result.shape()(1); ++col) {
            float max = -std::numeric_limits<float>::infinity();
            for (size_t i = 0; i != pool(0); ++i) {
                for (size_t j = 0; j != pool(1); ++j) {
                    max = std::max(max, a(row * pool(0) + i, col * pool(1) + j));
                }
            }
            result(row, col) = max;
        }
    }
}

void maxPoolDiff(
        const ConstTensorRef& a,
        const ConstTensorRef& poolResult,
        const ConstTensorRef& poolDiff,
        const Shape& pool,
        TensorRef&& result) {
    for (size_t row = 0; row != result.shape()(0); ++row) {
        const size_t rowPool = row / pool(0);
        for (size_t col = 0; col != result.shape()(1); ++col) {
            const size_t colPool = col / pool(1);
            if (a(row, col) == poolResult(rowPool, colPool)) {
                result(row, col) = poolDiff(rowPool, colPool);
            }
        }
    }
}
