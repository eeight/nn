cmake_minimum_required(VERSION 3.5)
enable_testing()

project(nn)

find_package(CBLAS REQUIRED)

include_directories(
    contrib/variant/include
    ${CBLAS_INCLUDE_DIRS})

add_subdirectory(contrib/variant)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror -fno-omit-frame-pointer -march=native")
set(CMAKE_CXX_STANDARD 17)

add_library(
    tensor
    ad.cc
    error.cc
    program.cc
    shape.cc
    tensor.cc
    tensor_value.cc)

add_library(
    nn
    loss.cc nn.cc)

add_library(
    mnist
    mnist.cc)

add_executable(main main.cc)
target_link_libraries(main nn mnist tensor ${CBLAS_LIBRARIES})

find_package(Boost COMPONENTS unit_test_framework REQUIRED)

add_executable(tensor_test tensor_test.cc)
target_link_libraries(tensor_test
    tensor
    mpark_variant
    ${CBLAS_LIBRARIES}
    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
add_test(tensor_test, tensor_test)
