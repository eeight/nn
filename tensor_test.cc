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

    BOOST_TEST(loss.cols() == 1);
    BOOST_TEST(loss.rows() == 1);
    BOOST_TEST(loss.eval()(0, 0) == 2.0);

}

#if 0
BOOST_AUTO_TEST_CASE(decreasing_loss) {
    auto x = t::newTensor(arma::ones<Matrix>(1, 1));
    auto y = t::newTensor(arma::zeros<Matrix>(1, 1));
    auto w = t::newTensor(arma::ones<Matrix>(1, 1));
    auto b = t::newTensor(arma::zeros<Matrix>(1, 1));
    std::vector<t::Tensor> params = {w, b};

    auto loss = t::pow(x * w + b - y, 2) + t::pow(w, 2);

    BOOST_TEST(loss.cols() == 1);
    BOOST_TEST(loss.rows() == 1);

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
#endif
