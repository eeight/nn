cmake_minimum_required(VERSION 3.5)
enable_testing()

project(nn)

find_package(Armadillo REQUIRED)

include_directories(
    contrib/variant/include
    ${Armadillo_INCLUDE_DIRS})

add_subdirectory(contrib/variant)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror")
set(CMAKE_CXX_STANDARD 17)

add_library(
    tensor
    ad.cc
    program.cc
    tensor.cc)

add_library(
    nn
    activation.cc loss.cc nn.cc)

add_library(
    mnist
    mnist.cc)

add_executable(main main.cc)
target_link_libraries(main nn mnist tensor ${ARMADILLO_LIBRARY})

find_package(Boost COMPONENTS unit_test_framework REQUIRED)

add_executable(tensor_test tensor_test.cc)
target_link_libraries(tensor_test
    tensor
    mpark_variant
    ${ARMADILLO_LIBRARY}
    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
add_test(tensor_test, tensor_test)
