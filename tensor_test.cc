#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "tensor.h"
#include "loss.h"

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(const_tensor_test) {
    const Matrix m = arma::randu<Matrix>(10, 10);
    const auto t = newConstTensor(m);
    BOOST_REQUIRE(arma::approx_equal(m, t.eval(), "absdiff", 1e-6));
}

BOOST_AUTO_TEST_CASE(loss_value) {
    auto x = newTensor(arma::ones<Matrix>(1, 1));
    auto y = newTensor(arma::zeros<Matrix>(1, 1));
    auto w = newTensor(arma::ones<Matrix>(1, 1));
    auto b = newTensor(arma::zeros<Matrix>(1, 1));
    std::vector<Tensor> params = {w, b};

    auto tmp = x * w + b - y;
    auto loss = tmp % tmp + w % w;

    BOOST_TEST(loss.shape().isScalar());
    BOOST_TEST(loss.eval()(0, 0) == 2.0);

}

BOOST_AUTO_TEST_CASE(decreasing_loss_scalar) {
    auto x = newTensor(arma::ones<Matrix>(1, 1));
    auto y = newTensor(arma::zeros<Matrix>(1, 1));
    auto w = newTensor(arma::ones<Matrix>(1, 1));
    auto b = newTensor(arma::zeros<Matrix>(1, 1));
    std::vector<Tensor> params = {w, b};

    auto loss = pow(x * w + b - y, 2) + pow(w, 2);

    BOOST_TEST(loss.shape().isScalar());

    const float eta = 0.05;

    float lossValue = loss.eval()(0, 0);
    for (size_t i = 0; i != 100; ++i) {
        auto partial = diff(loss, params);
        for (size_t j = 0; j != params.size(); ++j) {
            params[j] += -eta * partial[j];
        }

        const float nextLossValue = loss.eval()(0, 0);
        BOOST_TEST(nextLossValue < lossValue);
        lossValue = nextLossValue;
    }
}

BOOST_AUTO_TEST_CASE(decreasing_loss_matrix) {
    auto x = newTensor(2, 1);
    auto y = newTensor(2, 1);
    auto w = newTensor(arma::ones<Matrix>(2, 2));
    auto b = newTensor(arma::zeros<Matrix>(2, 1));
    std::vector<Tensor> params = {w, b};

    const Col xValue{{0, 1}};
    const Col yValue{{1, 0}};

    x = xValue;
    y = yValue;

    auto loss = sumSquares(w * x + b - y) + sumSquares(w);

    BOOST_TEST(loss.shape().isScalar());

    const float eta = 0.05;

    float lossValue = loss.eval()(0, 0);
    for (size_t i = 0; i != 100; ++i) {
        auto partial = diff(loss, params);
        for (size_t j = 0; j != params.size(); ++j) {
            params[j] += -eta * partial[j];
        }

        const float nextLossValue = loss.eval()(0, 0);
        BOOST_TEST(nextLossValue < lossValue);
        lossValue = nextLossValue;
    }
}

Tensor sigmoid(const Tensor& x) {
    const auto ones = newConstTensor(
            arma::ones<Matrix>(x.shape().rows, x.shape().cols));
    return ones / (ones + exp(-x));
}

BOOST_AUTO_TEST_CASE(decreasing_loss_matrix_with_activation) {
    auto x = newTensor(2, 1);
    auto y = newTensor(2, 1);
    auto w = newTensor(arma::ones<Matrix>(2, 2));
    auto b = newTensor(arma::zeros<Matrix>(2, 1));
    std::vector<Tensor> params = {w, b};

    const Col xValue{{0, 1}};
    const Col yValue{{1, 0}};

    x = xValue;
    y = yValue;

    auto loss = sumSquares(sigmoid(w * x + b) - y) + sumSquares(w);

    BOOST_TEST(loss.shape().isScalar());

    const float eta = 0.05;

    float lossValue = loss.eval()(0, 0);
    for (size_t i = 0; i != 100; ++i) {
        auto partial = diff(loss, params);
        for (size_t j = 0; j != params.size(); ++j) {
            params[j] += -eta * partial[j];
        }

        const float nextLossValue = loss.eval()(0, 0);
        BOOST_TEST(nextLossValue < lossValue);
        lossValue = nextLossValue;
    }
}
