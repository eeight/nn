#include "mnist.h"

#include <string>
#include <fstream>
#include <stdexcept>
#include <arpa/inet.h>

namespace mnist {

namespace {

int nextInt(std::istream& in) {
    int result;
    in.read(reinterpret_cast<char *>(&result), sizeof(result));
    return ntohl(result);
}

std::vector<int> readLabels(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open " + path);
    }
    in.exceptions(std::ios::badbit | std::ios::failbit);

    if (nextInt(in) != 2049) {
        throw std::runtime_error("Unexpected magic in labels file");
    }

    std::vector<int> labels(nextInt(in));
    for (int& label: labels) {
        label = in.get();
    }
    in.close();

    return labels;
}

std::vector<TensorValue> read(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open " + path);
    }
    in.exceptions(std::ios::badbit | std::ios::failbit);
    if (nextInt(in) != 2051) {
        throw std::runtime_error("Unexpected magic in labels file");
    }
    const size_t imagesNumber = nextInt(in);
    std::vector<TensorValue> images;
    const size_t rows = nextInt(in);
    const size_t cols = nextInt(in);
    std::vector<unsigned char> bytes(cols * rows);
    for (size_t i = 0; i != imagesNumber; ++i) {
        images.push_back(TensorValue::zeros({rows, cols}));
        in.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
        for (size_t j = 0; j != bytes.size(); ++j) {
            images.back().data()[j] = bytes[j] / 255.0;
        }
    }

    in.close();

    return images;
}

TensorValue oneHot(size_t size, size_t i) {
    auto v = TensorValue::zeros({size});
    v(i) = 1.0f;
    return v;
}

std::vector<Sample> zip(std::vector<TensorValue> xs, const std::vector<int>& ys) {
    std::vector<Sample> result;
    result.reserve(xs.size());

    for (size_t i = 0; i != xs.size(); ++i) {
        result.push_back({std::move(xs[i]), oneHot(10, ys[i])});
    }

    return result;
}

} // namespace

std::vector<Sample> readTest() {
    return zip(
            read("data/mnist/t10k-images-idx3-ubyte"),
            readLabels("data/mnist/t10k-labels-idx1-ubyte"));
}

std::vector<Sample> readTrain() {
    return zip(
            read("data/mnist/train-images-idx3-ubyte"),
            readLabels("data/mnist/train-labels-idx1-ubyte"));
}

} // namespace mnist
