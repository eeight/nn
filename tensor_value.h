#pragma once

#include "shape.h"
#include "types.h"

struct Conv2D { size_t padTop; size_t padBottom; size_t padLeft; size_t padRight; };

class TensorValue {
public:
    TensorValue(float x);
    TensorValue(Matrix m);
    TensorValue(Cube c);
    Shape shape() const;

    static TensorValue zeros(const Shape& shape);
    static TensorValue ones(const Shape& shape);
    static TensorValue randn(const Shape& shape, float stddev = 1.0f);

    float asScalar() const;
    const Matrix& asMatrix() const;
    const Cube& asCube() const;

private:
    mpark::variant<float, Matrix, Cube> value_;
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
        const TensorValue& x,
        float factor,
        TensorValue* y);

void tile(const TensorValue& x, const Shape& multipler, TensorValue* y);
void untile(const TensorValue& x, const Shape& multipler, TensorValue* y);
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
