#pragma once

#include "train.h"

#include <vector>

namespace mnist {

std::vector<Sample> readTest();
std::vector<Sample> readTrain();

} // namespace mnist
