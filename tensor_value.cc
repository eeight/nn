#include "tensor_value.h"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <random>
#include <stdexcept>
#include <iostream>

#include <cblas.h>

namespace {

template <class Op>
void zipWith(
        const TensorValue& x, const TensorValue& y, Op op, TensorValue* result) {
    const size_t size = x.shape().size();
    for (size_t i = 0; i != size; ++i) {
        result->data()[i] = op(x.data()[i], y.data()[i]);
    }
}

template <class Op>
void map(const TensorValue& x, Op op, TensorValue* result) {
    const size_t size = x.shape().size();

    for (size_t i = 0; i != size; ++i) {
        result->data()[i] = op(x.data()[i]);
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

TensorValue::TensorValue(float x) :
    data_(1, x), shape_{}
{}

TensorValue::TensorValue(Shape shape, std::vector<float> data) :
    data_(std::move(data)), shape_(std::move(shape))
{
    assert(!data_.empty());
    assert(data_.size() == shape_.size());
}

TensorValue TensorValue::zeros(const Shape& shape) {
    return TensorValue{shape, std::vector<float>(shape.size(), 0.0f)};
}

TensorValue TensorValue::ones(const Shape& shape) {
    return TensorValue{shape, std::vector<float>(shape.size(), 1.0f)};
}

TensorValue TensorValue::randn(const Shape& shape, float stddev) {
    return randomTensor(shape, std::normal_distribution<float>(0, stddev));
}

TensorValue TensorValue::randu(const Shape& shape) {
    return randomTensor(shape, std::uniform_real_distribution<float>(0, 1));
}

const float& TensorValue::operator()(size_t i) const {
    if (shape_.dim() != 1) {
        throw std::logic_error("Not a vector, shape = " + shape_.toString());
    }
    return data_.at(i);
}

const float& TensorValue::operator()(size_t i, size_t j) const {
    if (shape_.dim() != 2) {
        throw std::logic_error("Not a matrix, shape = " + shape_.toString());
    }
    return data_.at(i * shape_(0) + j);
}

const float& TensorValue::operator()(size_t i, size_t j, size_t k) const {
    if (shape_.dim() != 3) {
        throw std::logic_error("Not a cube, shape = " + shape_.toString());
    }
    return data_.at((i * shape_(0) + j) * shape_(1) + k);
}

const float& TensorValue::operator()(const std::vector<size_t> &indices) const {
    if (shape_.dim() != indices.size()) {
        throw std::logic_error(
                "Unexpected number of indices: " + std::to_string(indices.size()) +
                " for shape " + shape_.toString() );
    }
    if (indices.empty()) {
        return data_.front();
    }

    size_t index = indices.front();
    for (size_t i = 1; i != indices.size(); ++i) {
        index = index * shape_(i - 1) + indices[i];
    }

    return data_.at(index);
}

float& TensorValue::operator()(size_t i) {
    return const_cast<float &>(const_cast<const TensorValue &>(*this)(i));
}

float& TensorValue::operator()(size_t i, size_t j) {
    return const_cast<float &>(const_cast<const TensorValue &>(*this)(i, j));
}

float& TensorValue::operator()(size_t i, size_t j, size_t k) {
    return const_cast<float &>(const_cast<const TensorValue &>(*this)(i, j, k));
}

float& TensorValue::operator()(const std::vector<size_t> &indices) {
    return const_cast<float &>(const_cast<const TensorValue &>(*this)(indices));
}

float TensorValue::toScalar() const {
    if (shape_.size() != 1) {
        throw std::logic_error(
                "TensorValue::toScalar: cannot be converted to scalar");
    }
    return data_.front();
}

void multiply(
        const TensorValue& x,
        bool transposeX,
        const TensorValue& y,
        bool transposeY,
        bool negateResult,
        TensorValue* result) {
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
            result->data(),
            result->shape()(1));
}

void add(
        const TensorValue& x,
        bool transposeX,
        bool negateX,
        const TensorValue& y,
        bool transposeY,
        bool negateY,
        TensorValue* result)
{
    assert(!transposeX && !transposeY);
    if (negateX && negateY) {
        zipWith(x, y, [](float x, float y) { return -x - y; }, result);
    } else if (negateX && !negateY) {
        zipWith(x, y, [](float x, float y) { return y - x; }, result);
    } else if (!negateX && negateY) {
        zipWith(x, y, [](float x, float y) { return x - y; }, result);
    } else {
        zipWith(x, y, [](float x, float y) { return x + y; }, result);
    }
}

void divide(
        const TensorValue& x,
        bool transposeX,
        const TensorValue& y,
        bool transposeY,
        bool negateResult,
        TensorValue* result)
{
    assert(!transposeX && !transposeY);
    if (negateResult) {
        zipWith(x, y, [](float x, float y) { return -x / y; }, result);
    } else {
        zipWith(x, y, [](float x, float y) { return x / y; }, result);
    }
}

void hadamard(
        const TensorValue& x,
        bool transposeX,
        const TensorValue& y,
        bool transposeY,
        bool negateResult,
        TensorValue* result)
{
    assert(!transposeX && !transposeY);
    if (negateResult) {
        zipWith(x, y, [](float x, float y) { return -x * y; }, result);
    } else {
        zipWith(x, y, [](float x, float y) { return x * y; }, result);
    }
}

void pow(const TensorValue& x, float y, TensorValue* result) {
    map(x, [y](float x) { return std::pow(x, y); }, result);
}

void exp(const TensorValue& x, TensorValue* result) {
    map(x, [](float x) { return std::exp(x); }, result);
}

void log(const TensorValue& x, TensorValue* result) {
    map(x, [](float x) { return std::log(x); }, result);
}

void negate(const TensorValue& x, TensorValue* result) {
    map(x, [](float x) { return -x; }, result);
}

void transpose(const TensorValue& x, TensorValue* result) {
    const size_t rows = x.shape()(x.shape().dim() - 2);
    const size_t cols = x.shape()(x.shape().dim() - 1);
    for (size_t i = 0; i != rows; ++i) {
        for (size_t j = 0; j != cols; ++j) {
            (*result)(j, i) = x(i, j);
        }
    }
}

void reverse(const TensorValue& x, TensorValue* result) {
    const size_t size = x.shape().size();
    for (size_t i = 0; i != size; ++i) {
        result->data()[size - i - 1] = x.data()[i];
    }
}

void reshape(const TensorValue& x, TensorValue* result) {
    std::copy(x.data(), x.dataEnd(), result->data());
}

float accu(const TensorValue& x) {
    return std::accumulate(x.data(), x.dataEnd(), 0.0f);
}

void addMultiply(
        const TensorValue& x, float factor, TensorValue* y) {
    zipWith(x, *y, [factor](float x, float y) { return x * factor + y; }, y);
}

void tile(const TensorValue& x, const Shape& multiplier, TensorValue* y) {
    std::vector<size_t> indices(multiplier.dim(), 0);
    std::vector<size_t> xIndices(multiplier.dim());
    const size_t size = y->shape().size();
    for (size_t i = 0; i != size; ++i) {
        size_t j = 0;
        for (; indices[j] == y->shape()(j) - 1; ++j) {
            indices[j] = 0;
        }
        ++indices[j];
        for (size_t j = 0; j != xIndices.size(); ++j) {
            xIndices[j] = indices[j] % multiplier(j);
        }
        (*y)(indices) = x(xIndices);
    }
}

void untile(const TensorValue& x, const Shape& multiplier, TensorValue* y) {
    std::vector<size_t> indices(multiplier.dim(), 0);
    std::vector<size_t> yIndices(multiplier.dim());
    const size_t size = x.shape().size();
    std::fill(y->data(), y->dataEnd(), 0.0f);
    for (size_t i = 0; i != size; ++i) {
        size_t j = 0;
        for (; indices[j] == y->shape()(j) - 1; ++j) {
            indices[j] = 0;
        }
        ++indices[j];
        for (size_t j = 0; j != yIndices.size(); ++j) {
            yIndices[j] = indices[j] % multiplier(j);
        }
        (*y)(yIndices) += x(indices);
    }
}

void sigmoid(const TensorValue& x, TensorValue* result) {
    map(x, [](float x) { return 1.0f / (1.0f + std::exp(-x)); }, result);
}

void halfSumSquares(const TensorValue& x, TensorValue* result) {
    *result->data() = 0.5f * std::accumulate(
            x.data(),
            x.dataEnd(),
            0.0f,
            [](float x, float y) { return x + y * y; });
}

void conv2d(
        const TensorValue& a,
        const TensorValue& k,
        const Conv2D& conv,
        TensorValue* result) {
    const size_t kRows = k.shape()(0);
    const size_t kCols = k.shape()(1);

    for (size_t row = 0; row < result->shape()(0); ++row) {
        for (size_t col = 0; col < result->shape()(1); ++col) {
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
            (*result)(row, col) = dot(
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
        const TensorValue& a, const Shape& pool, TensorValue* result) {
    for (size_t row = 0; row != result->shape()(0); ++row) {
        for (size_t col = 0; col != result->shape()(1); ++col) {
            float max = -std::numeric_limits<float>::infinity();
            for (size_t i = 0; i != pool(0); ++i) {
                for (size_t j = 0; j != pool(1); ++j) {
                    max = std::max(max, a(row * pool(0) + i, col * pool(1) + j));
                }
            }
            (*result)(row, col) = max;
        }
    }
}

void maxPoolDiff(
        const TensorValue& a,
        const TensorValue& poolResult,
        const TensorValue& poolDiff,
        const Shape& pool,
        TensorValue* result) {
    for (size_t row = 0; row != result->shape()(0); ++row) {
        const size_t rowPool = row / pool(0);
        for (size_t col = 0; col != result->shape()(1); ++col) {
            const size_t colPool = col / pool(1);
            if (a(row, col) == poolResult(rowPool, colPool)) {
                (*result)(row, col) = poolDiff(rowPool, colPool);
            }
        }
    }
}
