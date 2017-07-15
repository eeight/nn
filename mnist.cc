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

std::vector<Col> read(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open " + path);
    }
    in.exceptions(std::ios::badbit | std::ios::failbit);
    if (nextInt(in) != 2051) {
        throw std::runtime_error("Unexpected magic in labels file");
    }
    std::vector<Col> images(nextInt(in));
    const size_t rows = nextInt(in);
    const size_t cols = nextInt(in);
    std::vector<unsigned char> bytes(cols * rows);
    for (auto& image: images) {
        image.set_size(rows * cols);
        in.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
        for (size_t i = 0; i != bytes.size(); ++i) {
            image(i) = bytes[i];
        }
        image /= 255.0;
    }

    in.close();

    return images;
}

std::vector<Sample> zip(std::vector<Col> xs, const std::vector<int>& ys) {
    std::vector<Sample> result;
    result.reserve(xs.size());

    for (size_t i = 0; i != xs.size(); ++i) {
        result.push_back({std::move(xs[i]), ys[i]});
    }

    return result;
}

} // namespace

std::vector<Sample> readTest() {
    return zip(
            read("/Users/eeight/git/nn/data/mnist/t10k-images-idx3-ubyte"),
            readLabels("/Users/eeight/git/nn/data/mnist/t10k-labels-idx1-ubyte"));
}

std::vector<Sample> readTrain() {
    return zip(
            read("/Users/eeight/git/nn/data/mnist/train-images-idx3-ubyte"),
            readLabels("/Users/eeight/git/nn/data/mnist/train-labels-idx1-ubyte"));
}

} // namespace mnist
