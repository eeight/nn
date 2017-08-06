#pragma once

#include "shape.h"

#include <vector>

struct Conv2D { size_t padTop; size_t padBottom; size_t padLeft; size_t padRight; };

class TensorValue {
public:
    /* implicit */ TensorValue(float x);
    TensorValue(Shape shape, std::vector<float> data);
    const Shape& shape() const { return shape_; }
    const float* data() const { return data_.data(); }
    const float* dataEnd() const { return data_.data() + shape_.size(); }
    float* data() { return data_.data(); }
    float* dataEnd() { return data_.data() + shape_.size(); }

    static TensorValue zeros(const Shape& shape);
    static TensorValue ones(const Shape& shape);
    static TensorValue randn(const Shape& shape, float stddev = 1.0f);
    static TensorValue randu(const Shape& shape);

    const float& operator()(size_t i) const;
    const float& operator()(size_t i, size_t j) const;
    const float& operator()(size_t i, size_t j, size_t k) const;
    const float& operator()(const std::vector<size_t> &indices) const;
    float& operator()(size_t i);
    float& operator()(size_t i, size_t j);
    float& operator()(size_t i, size_t j, size_t k);
    float& operator()(const std::vector<size_t> &indices);

    float toScalar() const;

private:
    std::vector<float> data_;
    Shape shape_;
};

void multiply(
        const TensorValue& x,
        bool transposeX,
        const TensorValue& y,
        bool transposeY,
        bool negateResult,
        TensorValue* result);

void add(
        const TensorValue& x,
        bool transposeX,
        bool negateX,
        const TensorValue& y,
        bool transposeY,
        bool negateY,
        TensorValue* result);

void divide(
        const TensorValue& x,
        bool transposeX,
        const TensorValue& y,
        bool transposeY,
        bool negateResult,
        TensorValue* result);

void hadamard(
        const TensorValue& x,
        bool transposeX,
        const TensorValue& y,
        bool transposeY,
        bool negateResult,
        TensorValue* result);

void pow(const TensorValue& x, float y, TensorValue* result);
void exp(const TensorValue& x, TensorValue* result);
void log(const TensorValue& x, TensorValue* result);
void negate(const TensorValue& x, TensorValue* result);
void transpose(const TensorValue& x, TensorValue* result);
void reverse(const TensorValue& x, TensorValue* result);
void reshape(const TensorValue& x, TensorValue* result);
float accu(const TensorValue& x);

// y += factor * x
void addMultiply(
        const TensorValue& x, float factor, TensorValue* y);

void tile(const TensorValue& x, const Shape& multiplier, TensorValue* y);
void untile(const TensorValue& x, const Shape& multiplier, TensorValue* y);
void sigmoid(const TensorValue& x, TensorValue* result);
void halfSumSquares(const TensorValue& x, TensorValue* result);
void conv2d(
        const TensorValue& a,
        const TensorValue& k,
        const Conv2D& conv,
        TensorValue* result);
void maxPool(
        const TensorValue& a, const Shape& pool, TensorValue* result);
void maxPoolDiff(
        const TensorValue& a,
        const TensorValue& poolResult,
        const TensorValue& poolDiff,
        const Shape& pool,
        TensorValue* result);
