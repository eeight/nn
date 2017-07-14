#pragma once

#include "types.h"

#include <vector>

namespace mnist {

std::vector<int> readTestLabels();
std::vector<Matrix> readTest();

std::vector<int> readTrainLabels();
std::vector<Matrix> readTrain();

} // namespace mnist
