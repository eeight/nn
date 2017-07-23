#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "ad.h"
#include "eval.h"
#include "tensor.h"

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(const_tensor) {
    const Matrix m = arma::randu<Matrix>(10, 10);
    const auto t = newConstTensor(m);
    BOOST_REQUIRE(arma::approx_equal(m, eval(t), "absdiff", 1e-6));
}

BOOST_AUTO_TEST_CASE(arg_passthrough) {
    const Matrix m = arma::randu<Matrix>(10, 10);
    const auto t = newTensor("x", Shape{m});
    BOOST_REQUIRE(arma::approx_equal(m, eval(t, {"x"}, {&m}), "absdiff", 1e-6));
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
    BOOST_TEST(eval(loss)(0, 0) == 2.0);

}

BOOST_AUTO_TEST_CASE(decreasing_loss_scalar) {
    auto x = newTensor(arma::ones<Matrix>(1, 1));
    auto y = newTensor(arma::zeros<Matrix>(1, 1));
    auto w = newTensor(arma::ones<Matrix>(1, 1));
    auto b = newTensor(arma::zeros<Matrix>(1, 1));
    std::vector<Tensor> params = {w, b};

    auto loss = pow(x * w + b - y, 2) + pow(w, 2);
    auto dLoss = compile(diff(loss, params), {});

    BOOST_TEST(loss.shape().isScalar());

    const float eta = 0.05;

    float lossValue = eval(loss)(0, 0);
    for (size_t i = 0; i != 100; ++i) {
        auto partial = dLoss();
        for (size_t j = 0; j != params.size(); ++j) {
            params[j] += -eta * partial[j];
        }

        const float nextLossValue = eval(loss)(0, 0);
        BOOST_TEST(nextLossValue < lossValue);
        lossValue = nextLossValue;
    }
}

BOOST_AUTO_TEST_CASE(decreasing_loss_matrix) {
    auto x = newTensor("x", 2, 1);
    auto y = newTensor("y", 2, 1);
    auto w = newTensor(arma::ones<Matrix>(2, 2));
    auto b = newTensor(arma::zeros<Matrix>(2, 1));
    std::vector<Tensor> params = {w, b};

    const Col xValue{{0, 1}};
    const Col yValue{{1, 0}};
    const std::vector<const Matrix *> args = {&xValue, &yValue};

    auto loss = sumSquares(w * x + b - y) + sumSquares(w);
    auto dLoss = compile(diff(loss, params), {"x", "y"});

    BOOST_TEST(loss.shape().isScalar());

    const float eta = 0.05;

    float lossValue = eval(loss, {"x", "y"}, args)(0, 0);
    for (size_t i = 0; i != 100; ++i) {
        auto partial = dLoss(args);
        for (size_t j = 0; j != params.size(); ++j) {
            params[j] += -eta * partial[j];
        }

        const float nextLossValue = eval(loss, {"x", "y"}, args)(0, 0);
        BOOST_TEST(nextLossValue < lossValue);
        lossValue = nextLossValue;
    }
}

BOOST_AUTO_TEST_CASE(decreasing_loss_matrix_with_activation) {
    auto x = newTensor(Col{{0, 1}});
    auto y = newTensor(Col{{1, 0}});
    auto w = newTensor(arma::ones<Matrix>(2, 2));
    auto b = newTensor(arma::zeros<Matrix>(2, 1));
    std::vector<Tensor> params = {w, b};

    auto lossTensor = sumSquares(sigmoid(w * x + b) - y) + sumSquares(w);
    BOOST_TEST(lossTensor.shape().isScalar());
    auto loss = compile({lossTensor}, {});
    auto dLoss = compile(diff(lossTensor, params), {});

    const float eta = 0.05;

    float lossValue = loss().front()(0, 0);
    for (size_t i = 0; i != 100; ++i) {
        auto partial = dLoss();
        for (size_t j = 0; j != params.size(); ++j) {
            params[j] += -eta * partial[j];
        }

        const float nextLossValue = loss().front()(0, 0);
        BOOST_TEST(nextLossValue < lossValue);
        lossValue = nextLossValue;
    }
}

BOOST_AUTO_TEST_CASE(sum_squares) {
    auto x = newTensor(Col{{0, 1}});
    auto y = newTensor(Col{{1, 0}});
    auto w = newTensor(arma::ones<Matrix>(2, 2));

    auto l = sumSquares(w * x - y) + sumSquares(w);
    auto dl = diff(l, {w});
    std::cerr << "L:\n" << compile({l}, {});
    std::cerr << "DL:\n" << compile(dl, {});
}
