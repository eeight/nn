#pragma once

#include "shape.h"

#include <vector>

struct Conv2D { size_t padTop; size_t padBottom; size_t padLeft; size_t padRight; };

namespace detail {

template <class Derived>
class ConstTensorBase {
public:
    explicit ConstTensorBase(Shape shape);
    const Shape& shape() const { return shape_; }
    const float* dataEnd() const { return getData() + shape_.size(); }

    const float& operator()(size_t i) const;
    const float& operator()(size_t i, size_t j) const;
    const float& operator()(size_t i, size_t j, size_t k) const;
    const float& operator()(const std::vector<size_t> &indices) const;

    float toScalar() const;

private:
    const float* getData() const {
        return static_cast<const Derived *>(this)->data();
    }

    Shape shape_;
};

template <class Derived>
class TensorBase : public ConstTensorBase<Derived> {
public:
    explicit TensorBase(Shape shape);
    float* dataEnd() { return getData() + this->shape().size(); }
    const float* dataEnd() const;

    const float& operator()(size_t i) const;
    const float& operator()(size_t i, size_t j) const;
    const float& operator()(size_t i, size_t j, size_t k) const;
    const float& operator()(const std::vector<size_t> &indices) const;
    float& operator()(size_t i);
    float& operator()(size_t i, size_t j);
    float& operator()(size_t i, size_t j, size_t k);
    float& operator()(const std::vector<size_t> &indices);

private:
    float* getData() { return static_cast<Derived *>(this)->data(); }
};

} // namespace detail

class TensorValue : public detail::TensorBase<TensorValue> {
public:
    /* implicit */ TensorValue(float x);
    explicit TensorValue(Shape shape, std::vector<float> data);
    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }

    static TensorValue zeros(const Shape& shape);
    static TensorValue ones(const Shape& shape);
    static TensorValue randn(const Shape& shape, float stddev = 1.0f);
    static TensorValue randu(const Shape& shape);

private:
    std::vector<float> data_;
};

class TensorRef;

class ConstTensorRef : public detail::ConstTensorBase<ConstTensorRef> {
public:
    /* implicit */ ConstTensorRef(const TensorValue& value);
    /* implicit */ ConstTensorRef(const TensorRef& ref);
    ConstTensorRef(Shape shape, const float* data);

    const float* data() const { return data_; }
    ConstTensorRef reshape(Shape shape) const;
private:
    const float* data_;
};

class TensorRef : public detail::TensorBase<TensorRef> {
public:
    /* implicit */ TensorRef(TensorValue& value);
    /* implicit */ TensorRef(TensorValue* value);

    const float* data() const { return data_; }
    float* data() { return data_; }

private:
    float* data_;
};

void multiply(
        const ConstTensorRef& x,
        bool transposeX,
        const ConstTensorRef& y,
        bool transposeY,
        bool negateResult,
        TensorRef&& result);

void add(
        const ConstTensorRef& x,
        bool transposeX,
        bool negateX,
        const ConstTensorRef& y,
        bool transposeY,
        bool negateY,
        TensorRef&& result);

void divide(
        const ConstTensorRef& x,
        bool transposeX,
        const ConstTensorRef& y,
        bool transposeY,
        bool negateResult,
        TensorRef&& result);

void hadamard(
        const ConstTensorRef& x,
        bool transposeX,
        const ConstTensorRef& y,
        bool transposeY,
        bool negateResult,
        TensorRef&& result);

void pow(const ConstTensorRef& x, float y, TensorRef&& result);
void exp(const ConstTensorRef& x, TensorRef&& result);
void log(const ConstTensorRef& x, TensorRef&& result);
void negate(const ConstTensorRef& x, TensorRef&& result);
void transpose(const ConstTensorRef& x, TensorRef&& result);
void reverse(const ConstTensorRef& x, TensorRef&& result);
void reshape(const ConstTensorRef& x, TensorRef&& result);
float accu(const ConstTensorRef& x);

// y += factor * x
void addMultiply(
        const ConstTensorRef& x, float factor, TensorRef&& y);

void tile(const ConstTensorRef& x, const Shape& multiplier, TensorRef&& y);
void untile(const ConstTensorRef& x, const Shape& multiplier, TensorRef&& y);
void sigmoid(const ConstTensorRef& x, TensorRef&& result);
void halfSumSquares(const ConstTensorRef& x, TensorRef&& result);
void conv2d(
        const ConstTensorRef& a,
        const ConstTensorRef& k,
        const Conv2D& conv,
        TensorRef&& result);
void maxPool(
        const ConstTensorRef& a, const Shape& pool, TensorRef&& result);
void maxPoolDiff(
        const ConstTensorRef& a,
        const ConstTensorRef& poolResult,
        const ConstTensorRef& poolDiff,
        const Shape& pool,
        TensorRef&& result);
