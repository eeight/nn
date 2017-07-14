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

std::vector<Matrix> read(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open " + path);
    }
    in.exceptions(std::ios::badbit | std::ios::failbit);
    if (nextInt(in) != 2051) {
        throw std::runtime_error("Unexpected magic in labels file");
    }
    std::vector<Matrix> images(nextInt(in));
    const size_t rows = nextInt(in);
    const size_t cols = nextInt(in);
    std::vector<unsigned char> bytes(cols * rows);
    for (auto& image: images) {
        image.set_size(rows, cols);
        in.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
        for (size_t i = 0; i != bytes.size(); ++i) {
            image(i) = bytes[i];
        }
        image = image.t();
    }

    return images;
}

} // namespace

std::vector<int> readTestLabels() {
    return readLabels("/Users/eeight/git/nn/data/mnist/t10k-labels-idx1-ubyte");
}

std::vector<Matrix> readTest() {
    return read("/Users/eeight/git/nn/data/mnist/t10k-images-idx3-ubyte");

}

std::vector<int> readTrainLabels() {
    return readLabels("/Users/eeight/git/nn/data/mnist/train-labels-idx1-ubyte");
}

std::vector<Matrix> readTrain() {
    return read("/Users/eeight/git/nn/data/mnist/train-images-idx3-ubyte");
}

} // namespace mnist
