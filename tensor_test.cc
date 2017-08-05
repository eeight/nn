#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "ad.h"
#include "eval.h"
#include "tensor.h"

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(const_tensor) {
    const Matrix m = arma::randu<Matrix>(10, 10);
    const auto t = newConstTensor(m);
    BOOST_REQUIRE(arma::approx_equal(
                m, eval(t).asMatrix(), "absdiff", 1e-6));
}

BOOST_AUTO_TEST_CASE(arg_passthrough) {
    const auto m = TensorValue::randu({10, 10});
    const auto t = newPlaceholder({10, 10});
    BOOST_REQUIRE(arma::approx_equal(
                m.asMatrix(), eval(t, {t}, {&m}).asMatrix(), "absdiff", 1e-6));
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

BOOST_AUTO_TEST_CASE(decreasing_loss_matrix) {
    auto x = newPlaceholder({2, 1});
    auto y = newPlaceholder({2, 1});
    auto w = newTensor(TensorValue::ones({2, 2}));
    auto b = newTensor(TensorValue::zeros({2, 1}));
    std::vector<Tensor> params = {w, b};

    const TensorValue xValue{Col{0, 1}};
    const TensorValue yValue{Col{1, 0}};
    const std::vector<const TensorValue *> args = {&xValue, &yValue};

    auto loss = halfSumSquares(w * x + b - y) + halfSumSquares(w);
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
    auto x = newTensor(Col{{0, 1}});
    auto y = newTensor(Col{{1, 0}});
    auto w = newTensor(TensorValue::ones({2, 2}));
    auto b = newTensor(TensorValue::zeros({2, 1}));
    std::vector<Tensor> params = {w, b};

    auto lossTensor = halfSumSquares(sigmoid(w * x + b) - y) + halfSumSquares(w);
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
    auto x = newTensor(Col{{0, 1}});
    auto y = newTensor(Col{{1, 0}});
    auto w = newTensor(TensorValue::ones({2, 2}));

    auto l = halfSumSquares(w * x - y) + halfSumSquares(w);
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
