#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "tensor.h"

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(const_tensor_test) {
    const Matrix m = arma::randu<Matrix>(10, 10);
    const auto t = t::newConstTensor(m);
    BOOST_REQUIRE(arma::approx_equal(m, t.eval(), "absdiff", 1e-6));
}

BOOST_AUTO_TEST_CASE(loss_value) {
    auto x = t::newTensor(arma::ones<Matrix>(1, 1));
    auto y = t::newTensor(arma::zeros<Matrix>(1, 1));
    auto w = t::newTensor(arma::ones<Matrix>(1, 1));
    auto b = t::newTensor(arma::zeros<Matrix>(1, 1));
    std::vector<t::Tensor> params = {w, b};

    auto tmp = x * w + b - y;
    auto loss = tmp % tmp + w % w;

    BOOST_TEST(loss.shape().isScalar());
    BOOST_TEST(loss.eval()(0, 0) == 2.0);

}

BOOST_AUTO_TEST_CASE(decreasing_loss_scalar) {
    auto x = t::newTensor(arma::ones<Matrix>(1, 1));
    auto y = t::newTensor(arma::zeros<Matrix>(1, 1));
    auto w = t::newTensor(arma::ones<Matrix>(1, 1));
    auto b = t::newTensor(arma::zeros<Matrix>(1, 1));
    std::vector<t::Tensor> params = {w, b};

    auto loss = t::pow(x * w + b - y, 2) + t::pow(w, 2);

    BOOST_TEST(loss.shape().isScalar());

    const float eta = 0.05;

    float lossValue = loss.eval()(0, 0);
    for (size_t i = 0; i != 100; ++i) {
        auto partial = t::diff(loss, params);
        for (size_t j = 0; j != params.size(); ++j) {
            params[j] += -eta * partial[j];
        }

        const float nextLossValue = loss.eval()(0, 0);
        BOOST_TEST(nextLossValue < lossValue);
        lossValue = nextLossValue;
    }
}

t::Tensor sumSquares(const t::Tensor& tensor) {
    size_t size = tensor.shape().size();
    return tensor.reshape({1, size}) * tensor.reshape({size, 1});
}

BOOST_AUTO_TEST_CASE(decreasing_loss_matrix) {
    auto x = t::newTensor(Col{{0, 1}});
    auto y = t::newTensor(Col{{1, 0}});
    auto w = t::newTensor(arma::ones<Matrix>(2, 2));
    auto b = t::newTensor(arma::zeros<Matrix>(2, 1));
    std::vector<t::Tensor> params = {w, b};

    auto loss = sumSquares(w * x + b - y) + sumSquares(w);

    BOOST_TEST(loss.shape().isScalar());

    const float eta = 0.05;

    float lossValue = loss.eval()(0, 0);
    for (size_t i = 0; i != 100; ++i) {
        auto partial = t::diff(loss, params);
        for (size_t j = 0; j != params.size(); ++j) {
            params[j] += -eta * partial[j];
        }

        const float nextLossValue = loss.eval()(0, 0);
        BOOST_TEST(nextLossValue < lossValue);
        lossValue = nextLossValue;
    }
}
