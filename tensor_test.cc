#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "ad.h"
#include "eval.h"
#include "tensor.h"

#include <boost/test/unit_test.hpp>

bool approxEqual(const TensorValue& x, const TensorValue& y) {
    if (x.shape() != y.shape()) {
        return false;
    }
    for (size_t i = 0; i != x.shape().size(); ++i) {
        if (std::abs(x.data()[i] - y.data()[i]) > 1e-6) {
            return false;
        }
    }

    return true;
}

TensorValue row(std::initializer_list<float> values) {
    auto x = TensorValue::zeros({1, values.size()});
    std::copy(values.begin(), values.end(), x.data());
    return x;
}
#if 0

BOOST_AUTO_TEST_CASE(const_tensor) {
    const auto m = TensorValue::randu({10, 10});
    const auto t = newConstTensor(m);
    BOOST_REQUIRE(approxEqual(m, eval(t)));
}

BOOST_AUTO_TEST_CASE(arg_passthrough) {
    const auto m = TensorValue::randu({10, 10});
    const auto t = newPlaceholder({10, 10});
    BOOST_REQUIRE(approxEqual(m, eval(t, {t}, {&m})));
}

BOOST_AUTO_TEST_CASE(scalar_loss_value) {
    auto x = newTensor(TensorValue::ones({}));
    auto y = newTensor(TensorValue::zeros({}));
    auto w = newTensor(TensorValue::ones({}));
    auto b = newTensor(TensorValue::zeros({}));
    std::vector<Tensor> params = {w, b};

    auto tmp = x * w + b - y;
    auto loss = tmp % tmp + w % w;

    BOOST_TEST(loss.shape().size() == 1);
    BOOST_TEST(eval(loss).toScalar() == 2.0);
}

BOOST_AUTO_TEST_CASE(decreasing_loss_scalar) {
    auto x = newTensor(TensorValue::ones({}));
    auto y = newTensor(TensorValue::zeros({}));
    auto w = newTensor(TensorValue::ones({}));
    auto b = newTensor(TensorValue::zeros({}));
    std::vector<Tensor> params = {w, b};

    auto loss = pow(x * w + b - y, 2) + pow(w, 2);
    auto dLoss = compile(diff(loss, params), {});
    std::cerr << dLoss << '\n';

    BOOST_TEST(loss.shape().size() == 1);

    const float eta = 0.05;

    float lossValue = eval(loss).toScalar();
    for (size_t i = 0; i != 100; ++i) {
        auto partial = dLoss();
        for (size_t j = 0; j != params.size(); ++j) {
            mutate(params[j], [&](TensorValue& param) {
                addMultiply(partial[j], -eta, &param);
            });
        }

        const float nextLossValue = eval(loss).toScalar();
        BOOST_TEST(nextLossValue < lossValue);
        lossValue = nextLossValue;
    }
}
#endif

BOOST_AUTO_TEST_CASE(decreasing_loss_matrix) {
    auto x = newPlaceholder({1, 2});
    auto y = newPlaceholder({1, 2});
    auto w = newTensor(TensorValue::ones({2, 2}));
    auto b = newTensor(TensorValue::zeros({1, 2}));
    std::vector<Tensor> params = {w, b};

    const auto xValue = row({0, 1});
    const auto yValue = row({1, 0});
    const std::vector<const TensorValue *> args = {&xValue, &yValue};

    auto loss = halfSumSquares(x * w + b - y) + halfSumSquares(w);
    auto dLoss = compile(diff(loss, params), {x, y});

    BOOST_TEST(loss.shape().size() == 1);

    const float eta = 0.05;

    float lossValue = eval(loss, {x, y}, args).toScalar();
    for (size_t i = 0; i != 100; ++i) {
        auto partial = dLoss(args);
        for (size_t j = 0; j != params.size(); ++j) {
            mutate(params[j], [&](TensorValue& param) {
                addMultiply(partial[j], -eta, &param);
            });
        }

        const float nextLossValue = eval(loss, {x, y}, args).toScalar();
        BOOST_TEST(nextLossValue < lossValue);
        lossValue = nextLossValue;
    }
}

BOOST_AUTO_TEST_CASE(decreasing_loss_matrix_with_activation) {
    auto x = newTensor(row({0, 1}));
    auto y = newTensor(row({1, 0}));
    auto w = newTensor(TensorValue::ones({2, 2}));
    auto b = newTensor(TensorValue::zeros({1, 2}));
    std::vector<Tensor> params = {w, b};

    auto lossTensor = halfSumSquares(sigmoid(x * w + b) - y) + halfSumSquares(w);
    BOOST_TEST(lossTensor.shape().size() == 1);
    auto loss = compile({lossTensor}, {});
    auto dLoss = compile(diff(lossTensor, params), {});

    const float eta = 0.05;

    float lossValue = loss().front().toScalar();
    for (size_t i = 0; i != 100; ++i) {
        auto partial = dLoss();
        for (size_t j = 0; j != params.size(); ++j) {
            mutate(params[j], [&](TensorValue& param) {
                addMultiply(partial[j], -eta, &param);
            });
        }

        const float nextLossValue = loss().front().toScalar();
        BOOST_TEST(nextLossValue < lossValue);
        lossValue = nextLossValue;
    }
}

BOOST_AUTO_TEST_CASE(sum_squares) {
    auto x = newTensor(row({0, 1}));
    auto y = newTensor(row({1, 0}));
    auto w = newTensor(TensorValue::ones({2, 2}));

    auto l = halfSumSquares(x * w - y) + halfSumSquares(w);
    auto dl = diff(l, {w});
    std::cerr << "L:\n" << compile({l}, {});
    std::cerr << "DL:\n" << compile(dl, {});
}

BOOST_AUTO_TEST_CASE(convolution) {
    auto a = newTensor(TensorValue::randu({5, 5}));
    auto k = newTensor(TensorValue::randu({2, 2}));
    auto t = newTensor(TensorValue::randu({2, 2}));

    auto loss = halfSumSquares(
            maxPool(conv2d(a, k, /* sameSize = */ false), {2, 2}) - t);
    auto dLoss = compile(diff(loss, {k}), {});

    const float eta = 0.05;

    float lossValue = eval(loss).toScalar();
    const size_t n = 50;
    size_t improvements = 0;
    for (size_t i = 0; i != n; ++i) {
        auto partial = dLoss();
        mutate(k, [&](TensorValue& k) {
            addMultiply(partial[0], -eta, &k);
        });

        const float nextLossValue = eval(loss).toScalar();
        improvements += nextLossValue < lossValue;
        lossValue = nextLossValue;
    }

    BOOST_TEST(improvements > 0.8 * n);
}

BOOST_AUTO_TEST_CASE(tile_test) {
    auto x = TensorValue::zeros({1, 2});
    x(0, 0) = 1;
    x(0, 1) = 2;

    auto y = TensorValue::zeros({3, 4});
    tile(x, {3, 2}, &y);
    for (size_t i = 0; i != 3; ++i) {
        BOOST_TEST(y(i, 0) == 1);
        BOOST_TEST(y(i, 1) == 2);
        BOOST_TEST(y(i, 2) == 1);
        BOOST_TEST(y(i, 3) == 2);
    }
    untile(y, {3, 2}, &x);
    BOOST_TEST(x(0, 0) == 6);
    BOOST_TEST(x(0, 1) == 12);
}
